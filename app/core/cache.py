"""
Cache module for PyHellen API.

Two tiers: an in-memory LRU (bounded by entries and JSON bytes) in front of SQLite persistence.
Keys cover everything that can change a tagging result: a model fingerprint (library and model
versions, device, quantization), the model name and the text as actually tagged.
SQLite work and JSON (de)serialization run in a worker thread so they never block the event loop.
"""

import asyncio
import contextlib
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from app.core.logger import logger


@dataclass
class CacheEntry:
    """A single cache entry with value and expiration time."""

    value: Any
    expires_at: float
    model: str
    hits: int = 0
    size_bytes: int = 0


def _default_fingerprint(model: str) -> str:
    from app.core.utils import model_fingerprint

    return model_fingerprint(model)


class HybridCache:
    """
    Hybrid LRU Cache with in-memory speed and SQLite persistence.

    Features:
    - In-memory LRU bounded by entry count and total JSON size
    - SQLite persistence (off the event loop) for recovery after restart
    - Versioned keys: results are never reused across model/library versions or inference settings
    - Bulk get/set for batch endpoints, single-flight computation for concurrent identical requests
    - TTL with lazy expiration plus an optional periodic purge
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 3600,
        persist: bool = True,
        *,
        enabled: bool = True,
        max_bytes: Optional[int] = None,
        db_max_size: Optional[int] = None,
        db_max_bytes: Optional[int] = None,
        max_entry_bytes: Optional[int] = None,
        fingerprint: Optional[Callable[[str], str]] = None,
        store_text_preview: bool = True,
    ):
        """
        Initialize the hybrid cache.

        Args:
            max_size: Maximum number of entries in memory
            ttl_seconds: Time-to-live in seconds for each entry
            persist: Whether to persist to SQLite (can be disabled for testing)
            enabled: When False, nothing is cached and every lookup misses
            max_bytes: Maximum total JSON size of in-memory entries (None = unbounded)
            db_max_size: Maximum number of entries in SQLite (defaults to 2 * max_size)
            db_max_bytes: Maximum total JSON size in SQLite (None = unbounded)
            max_entry_bytes: Results whose JSON is larger are not cached (None = no limit)
            fingerprint: Callable returning a version fingerprint for a model name, part of every key
            store_text_preview: Store the first 100 characters of texts in SQLite
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._max_bytes = max_bytes
        self._ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._enabled = enabled
        self._persist = persist and enabled
        self._db_max_size = db_max_size or max_size * 2
        self._db_max_bytes = db_max_bytes
        self._max_entry_bytes = max_entry_bytes
        self._fingerprint = fingerprint
        self._store_text_preview = store_text_preview
        self._memory_bytes = 0
        self._inflight: Dict[str, asyncio.Future] = {}
        self._db_repo = None

    @classmethod
    def from_settings(cls, settings) -> "HybridCache":
        """Build the application cache from Settings (CACHE_* environment variables)."""
        return cls(
            max_size=settings.cache_memory_max_entries,
            ttl_seconds=settings.cache_ttl_seconds,
            persist=settings.cache_persist,
            enabled=settings.cache_enabled,
            max_bytes=settings.cache_memory_max_bytes,
            db_max_size=settings.cache_db_max_entries,
            db_max_bytes=settings.cache_db_max_bytes,
            max_entry_bytes=settings.cache_max_entry_bytes,
            fingerprint=_default_fingerprint,
            store_text_preview=settings.cache_store_text_preview,
        )

    def _get_repo(self):
        """Lazy initialization of database repository."""
        if self._db_repo is None and self._persist:
            try:
                from app.core.database import CacheRepository

                self._db_repo = CacheRepository(
                    max_size=self._db_max_size,
                    ttl_seconds=self._ttl_seconds,
                    max_bytes=self._db_max_bytes,
                )
            except Exception as e:
                logger.warning(f"Could not initialize cache persistence: {e}")
                self._persist = False
        return self._db_repo

    def _generate_key(self, model: str, text: str, lower: bool) -> str:
        """Key on the text as actually tagged: process_text lowercases it first when lower=True."""
        tagged_text = text.lower() if lower else text
        fingerprint = self._fingerprint(model) if self._fingerprint else ""
        return hashlib.sha256(f"{fingerprint}\0{model}\0{tagged_text}".encode()).hexdigest()[:32]

    # ===================
    # In-memory tier (callers hold self._lock)
    # ===================

    def _remove(self, key: str) -> None:
        entry = self._cache.pop(key, None)
        if entry is not None:
            self._memory_bytes -= entry.size_bytes

    def _store(self, key: str, entry: CacheEntry) -> None:
        self._remove(key)
        self._cache[key] = entry
        self._memory_bytes += entry.size_bytes
        while self._cache and (
            len(self._cache) > self._max_size or (self._max_bytes is not None and self._memory_bytes > self._max_bytes)
        ):
            oldest_key = next(iter(self._cache))
            self._remove(oldest_key)

    # ===================
    # Lookups
    # ===================

    async def get(self, model: str, text: str, lower: bool) -> Optional[Any]:
        """
        Get a value from the cache.

        First checks in-memory cache, then falls back to database.
        """
        return (await self.get_many(model, [text], lower))[0]

    async def get_many(self, model: str, texts: Sequence[str], lower: bool) -> List[Optional[Any]]:
        """Look up several texts at once (one database round-trip). Results keep the order of `texts`."""
        results: List[Optional[Any]] = [None] * len(texts)
        if not self._enabled or not texts:
            return results

        keys = [self._generate_key(model, text, lower) for text in texts]
        missing: Dict[str, List[int]] = {}
        now = time.time()
        async with self._lock:
            for index, key in enumerate(keys):
                entry = self._cache.get(key)
                if entry is not None and now > entry.expires_at:
                    self._remove(key)
                    entry = None
                if entry is None:
                    missing.setdefault(key, []).append(index)
                    continue
                self._cache.move_to_end(key)
                entry.hits += 1
                results[index] = entry.value

        repo = self._get_repo() if missing else None
        if repo:
            try:
                found = await asyncio.to_thread(self._load_from_db, repo, list(missing))
            except Exception as e:
                logger.warning(f"Error loading from cache DB: {e}")
                found = {}
            async with self._lock:
                for key, (value, expires_at, size_bytes) in found.items():
                    # Keep the database expiry: promotion must not extend an entry's lifetime
                    self._store(key, CacheEntry(value, expires_at, model, hits=1, size_bytes=size_bytes))
                    for index in missing[key]:
                        results[index] = value

        async with self._lock:
            hits = sum(result is not None for result in results)
            self._hits += hits
            self._misses += len(results) - hits
        return results

    @staticmethod
    def _load_from_db(repo, keys: List[str]) -> Dict[str, Tuple[Any, float, int]]:
        return {
            key: (json.loads(result_json), expires_at, len(result_json))
            for key, (result_json, expires_at) in repo.get_many(keys).items()
        }

    # ===================
    # Writes
    # ===================

    async def set(self, model: str, text: str, lower: bool, value: Any) -> None:
        """
        Set a value in the cache.

        Saves to both in-memory cache and database.
        """
        await self.set_many(model, [(text, value)], lower)

    async def set_many(self, model: str, items: Sequence[Tuple[str, Any]], lower: bool) -> None:
        """Store several (text, result) pairs at once (one database transaction)."""
        if not self._enabled or not items:
            return

        entries = {self._generate_key(model, text, lower): (text, value) for text, value in items}
        expires_at = time.time() + self._ttl_seconds
        repo = self._get_repo()
        stored = await asyncio.to_thread(self._serialize_and_persist, repo, model, entries, expires_at)

        async with self._lock:
            for key, (value, size_bytes) in stored.items():
                self._store(key, CacheEntry(value, expires_at, model, size_bytes=size_bytes))

    def _serialize_and_persist(
        self, repo, model: str, entries: Dict[str, Tuple[str, Any]], expires_at: float
    ) -> Dict[str, Tuple[Any, int]]:
        """Runs in a worker thread: serialize, apply the per-entry size limit, write to SQLite."""
        from app.core.database.repositories.cache_repo import CacheRecord

        stored: Dict[str, Tuple[Any, int]] = {}
        records = []
        for key, (text, value) in entries.items():
            result_json = json.dumps(value)
            if self._max_entry_bytes is not None and len(result_json) > self._max_entry_bytes:
                continue
            stored[key] = (value, len(result_json))
            records.append(
                CacheRecord(
                    key=key,
                    result_json=result_json,
                    expires_at=expires_at,
                    text_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
                    text_preview=text[:100] if self._store_text_preview else None,
                )
            )

        if repo and records:
            try:
                repo.set_many(model, records)
            except Exception as e:
                logger.warning(f"Error saving to cache DB: {e}")
        return stored

    async def get_or_compute(
        self, model: str, text: str, lower: bool, compute: Callable[[], Awaitable[Any]]
    ) -> Tuple[Any, bool]:
        """
        Return (result, from_cache), computing and caching the result on a miss.

        Concurrent requests for the same key share a single computation; the waiting requests
        report from_cache=True. Errors propagate to every waiter and nothing is cached.
        """
        if not self._enabled:
            return await compute(), False

        value = await self.get(model, text, lower)
        if value is not None:
            return value, True

        key = self._generate_key(model, text, lower)
        pending = self._inflight.get(key)
        if pending is not None:
            return await asyncio.shield(pending), True

        future = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            value = await compute()
            await self.set(model, text, lower, value)
            future.set_result(value)
            return value, False
        except asyncio.CancelledError:
            future.cancel()
            raise
        except Exception as e:
            future.set_exception(e)
            future.exception()  # Mark as retrieved when nobody is waiting
            raise
        finally:
            self._inflight.pop(key, None)

    # ===================
    # Maintenance
    # ===================

    async def _run_db(self, operation: str, *args) -> int:
        repo = self._get_repo()
        if not repo:
            return 0
        try:
            return await asyncio.to_thread(getattr(repo, operation), *args)
        except Exception as e:
            logger.warning(f"Cache DB {operation} failed: {e}")
            return 0

    async def clear(self) -> int:
        """Clear all entries from the cache. Returns number of distinct cleared entries."""
        async with self._lock:
            memory_count = len(self._cache)
            self._cache.clear()
            self._memory_bytes = 0
            self._hits = 0
            self._misses = 0

        count = max(memory_count, await self._run_db("clear"))
        logger.info(f"Cache cleared: {count} entries removed")
        return count

    async def cleanup_expired(self) -> int:
        """Remove expired entries. Returns number of removed entries."""
        async with self._lock:
            now = time.time()
            expired_keys = [key for key, entry in self._cache.items() if now > entry.expires_at]
            for key in expired_keys:
                self._remove(key)

        count = max(len(expired_keys), await self._run_db("cleanup_expired"))
        if count > 0:
            logger.info(f"Cache cleanup: {count} expired entries removed")
        return count

    async def clear_model(self, model: str) -> int:
        """Clear all cache entries for a specific model. Returns number of distinct cleared entries."""
        async with self._lock:
            keys_to_remove = [key for key, entry in self._cache.items() if entry.model == model]
            for key in keys_to_remove:
                self._remove(key)

        count = max(len(keys_to_remove), await self._run_db("clear_by_model", model))
        if count > 0:
            logger.info(f"Cleared {count} cache entries for model '{model}'")
        return count

    def start_cleanup_task(self, interval_seconds: float) -> asyncio.Task:
        """Start a background task purging expired entries every `interval_seconds`."""

        async def purge_periodically():
            while True:
                await asyncio.sleep(interval_seconds)
                with contextlib.suppress(Exception):
                    await self.cleanup_expired()

        return asyncio.create_task(purge_periodically())

    # ===================
    # Statistics
    # ===================

    @property
    def stats(self) -> Dict[str, Any]:
        """In-memory cache statistics (no database access)."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "enabled": self._enabled,
            "memory_size": len(self._cache),
            "memory_bytes": self._memory_bytes,
            "max_size": self._max_size,
            "max_bytes": self._max_bytes,
            "ttl_seconds": self._ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
            "persistence_enabled": self._persist,
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Statistics including the SQLite tier."""
        stats = self.stats
        repo = self._get_repo()
        if repo:
            try:
                stats["database"] = await asyncio.to_thread(repo.get_statistics)
            except Exception as e:
                logger.warning(f"Error reading cache DB statistics: {e}")
        return stats


def _build_default_cache() -> HybridCache:
    from app.core.settings import settings

    return HybridCache.from_settings(settings)


# Global cache instance, configured from CACHE_* settings
cache = _build_default_cache()


# Legacy alias for backwards compatibility
LRUCache = HybridCache
