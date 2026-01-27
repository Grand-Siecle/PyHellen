"""
Cache module for PyHellen API.
Implements an LRU cache with TTL for storing tagging results.
"""

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from app.core.logger import logger


@dataclass
class CacheEntry:
    """A single cache entry with value and expiration time."""
    value: Any
    expires_at: float
    hits: int = 0


class LRUCache:
    """
    Thread-safe LRU Cache with TTL support.

    Features:
    - LRU eviction policy
    - Time-to-live (TTL) for entries
    - Maximum size limit
    - Hit/miss statistics
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of entries in the cache
            ttl_seconds: Time-to-live in seconds for each entry
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    def _generate_key(self, model: str, text: str, lower: bool) -> str:
        """Generate a unique cache key from the input parameters."""
        content = f"{model}:{text}:{lower}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    async def get(self, model: str, text: str, lower: bool) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            model: Model name
            text: Input text
            lower: Lowercase flag

        Returns:
            Cached value if found and not expired, None otherwise
        """
        key = self._generate_key(model, text, lower)

        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]

                # Check if expired
                if time.time() > entry.expires_at:
                    del self._cache[key]
                    self._misses += 1
                    return None

                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.hits += 1
                self._hits += 1
                return entry.value

            self._misses += 1
            return None

    async def set(self, model: str, text: str, lower: bool, value: Any) -> None:
        """
        Set a value in the cache.

        Args:
            model: Model name
            text: Input text
            lower: Lowercase flag
            value: Value to cache
        """
        key = self._generate_key(model, text, lower)
        expires_at = time.time() + self._ttl_seconds

        async with self._lock:
            # Remove oldest if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)
            self._cache.move_to_end(key)

    async def clear(self) -> int:
        """Clear all entries from the cache. Returns number of cleared entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info(f"Cache cleared: {count} entries removed")
            return count

    async def cleanup_expired(self) -> int:
        """Remove expired entries. Returns number of removed entries."""
        async with self._lock:
            now = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if now > entry.expires_at
            ]
            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                logger.info(f"Cache cleanup: {len(expired_keys)} expired entries removed")
            return len(expired_keys)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2)
        }


# Global cache instance
cache = LRUCache(max_size=1000, ttl_seconds=3600)
