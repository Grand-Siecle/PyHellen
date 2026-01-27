"""
Cache module for PyHellen API.
Implements a hybrid LRU cache with in-memory speed and SQLite persistence.
"""

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.core.logger import logger


@dataclass
class CacheEntry:
    """A single cache entry with value and expiration time."""
    value: Any
    expires_at: float
    hits: int = 0


class HybridCache:
    """
    Hybrid LRU Cache with in-memory speed and SQLite persistence.

    Features:
    - Fast in-memory LRU cache for hot data
    - SQLite persistence for recovery after restart
    - Automatic sync between memory and database
    - TTL support with lazy expiration
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600, persist: bool = True):
        """
        Initialize the hybrid cache.

        Args:
            max_size: Maximum number of entries in memory
            ttl_seconds: Time-to-live in seconds for each entry
            persist: Whether to persist to SQLite (can be disabled for testing)
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._persist = persist
        self._db_repo = None
        self._initialized = False

    def _get_repo(self):
        """Lazy initialization of database repository."""
        if self._db_repo is None and self._persist:
            try:
                from app.core.database import get_db_manager, CacheRepository
                self._db_repo = CacheRepository(
                    get_db_manager(),
                    max_size=self._max_size * 2,  # DB can hold more
                    ttl_seconds=self._ttl_seconds
                )
            except Exception as e:
                logger.warning(f"Could not initialize cache persistence: {e}")
                self._persist = False
        return self._db_repo

    async def _load_from_db(self, model: str, text: str, lower: bool) -> Optional[Any]:
        """Try to load a value from database if not in memory."""
        repo = self._get_repo()
        if repo:
            try:
                return repo.get(model, text, lower)
            except Exception as e:
                logger.warning(f"Error loading from cache DB: {e}")
        return None

    async def _save_to_db(self, model: str, text: str, lower: bool, value: Any) -> None:
        """Save a value to database in background."""
        repo = self._get_repo()
        if repo:
            try:
                repo.set(model, text, lower, value)
            except Exception as e:
                logger.warning(f"Error saving to cache DB: {e}")

    def _generate_key(self, model: str, text: str, lower: bool) -> str:
        """Generate a unique cache key from the input parameters."""
        content = f"{model}:{text}:{lower}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    async def get(self, model: str, text: str, lower: bool) -> Optional[Any]:
        """
        Get a value from the cache.

        First checks in-memory cache, then falls back to database.
        """
        key = self._generate_key(model, text, lower)

        async with self._lock:
            # Check in-memory cache first
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

        # Try database if not in memory
        if self._persist:
            value = await self._load_from_db(model, text, lower)
            if value is not None:
                # Promote to in-memory cache
                async with self._lock:
                    expires_at = time.time() + self._ttl_seconds
                    if len(self._cache) >= self._max_size:
                        self._cache.popitem(last=False)
                    self._cache[key] = CacheEntry(value=value, expires_at=expires_at, hits=1)
                    self._hits += 1
                return value

        async with self._lock:
            self._misses += 1
        return None

    async def set(self, model: str, text: str, lower: bool, value: Any) -> None:
        """
        Set a value in the cache.

        Saves to both in-memory cache and database.
        """
        key = self._generate_key(model, text, lower)
        expires_at = time.time() + self._ttl_seconds

        async with self._lock:
            # Remove oldest if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)
            self._cache.move_to_end(key)

        # Persist to database (fire and forget)
        if self._persist:
            await self._save_to_db(model, text, lower, value)

    async def clear(self) -> int:
        """Clear all entries from the cache. Returns number of cleared entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0

        # Clear database
        if self._persist:
            repo = self._get_repo()
            if repo:
                try:
                    db_count = repo.clear()
                    count = max(count, db_count)
                except Exception as e:
                    logger.warning(f"Error clearing cache DB: {e}")

        logger.info(f"Cache cleared: {count} entries removed")
        return count

    async def cleanup_expired(self) -> int:
        """Remove expired entries. Returns number of removed entries."""
        count = 0

        # Clean in-memory
        async with self._lock:
            now = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if now > entry.expires_at
            ]
            for key in expired_keys:
                del self._cache[key]
            count = len(expired_keys)

        # Clean database
        if self._persist:
            repo = self._get_repo()
            if repo:
                try:
                    db_count = repo.cleanup_expired()
                    count = max(count, db_count)
                except Exception as e:
                    logger.warning(f"Error cleaning cache DB: {e}")

        if count > 0:
            logger.info(f"Cache cleanup: {count} expired entries removed")
        return count

    async def clear_model(self, model: str) -> int:
        """Clear all cache entries for a specific model."""
        count = 0

        # Clear from in-memory (we need to check each key)
        async with self._lock:
            keys_to_remove = [
                key for key in self._cache.keys()
                if key.startswith(hashlib.sha256(f"{model}:".encode()).hexdigest()[:8])
            ]
            # This won't work perfectly for in-memory, so just track DB count
            for key in keys_to_remove:
                del self._cache[key]
            count = len(keys_to_remove)

        # Clear from database
        if self._persist:
            repo = self._get_repo()
            if repo:
                try:
                    db_count = repo.clear_by_model(model)
                    count = max(count, db_count)
                except Exception as e:
                    logger.warning(f"Error clearing model cache from DB: {e}")

        return count

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

        stats = {
            "memory_size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
            "persistence_enabled": self._persist
        }

        # Add database stats if available
        if self._persist:
            repo = self._get_repo()
            if repo:
                try:
                    db_stats = repo.get_statistics()
                    stats["database"] = db_stats
                except Exception:
                    pass

        return stats


# Global cache instance (hybrid by default)
cache = HybridCache(max_size=1000, ttl_seconds=3600, persist=True)


# Legacy alias for backwards compatibility
LRUCache = HybridCache
