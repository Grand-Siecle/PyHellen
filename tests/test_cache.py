"""
Tests for the cache module.
"""

import asyncio
import pytest
import time

from app.core.cache import LRUCache, CacheEntry


class TestLRUCache:
    """Test suite for LRUCache."""

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, cache):
        """Test basic set and get operations."""
        await cache.set("model1", "test text", False, {"result": "data"})
        result = await cache.get("model1", "test text", False)
        assert result == {"result": "data"}

    @pytest.mark.asyncio
    async def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = await cache.get("nonexistent", "text", False)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_key_uniqueness(self, cache):
        """Test that different parameters produce different cache keys."""
        await cache.set("model1", "text", False, {"lower": False})
        await cache.set("model1", "text", True, {"lower": True})

        result1 = await cache.get("model1", "text", False)
        result2 = await cache.get("model1", "text", True)

        assert result1 == {"lower": False}
        assert result2 == {"lower": True}

    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test that expired entries are not returned."""
        short_ttl_cache = LRUCache(max_size=10, ttl_seconds=1)
        await short_ttl_cache.set("model", "text", False, "value")

        # Should be in cache immediately
        result = await short_ttl_cache.get("model", "text", False)
        assert result == "value"

        # Wait for expiration
        await asyncio.sleep(1.5)

        # Should be expired now
        result = await short_ttl_cache.get("model", "text", False)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        small_cache = LRUCache(max_size=3, ttl_seconds=3600)

        await small_cache.set("m", "text1", False, "value1")
        await small_cache.set("m", "text2", False, "value2")
        await small_cache.set("m", "text3", False, "value3")

        # Cache should be full
        assert small_cache.stats["memory_size"] == 3

        # Access text1 to make it recently used
        await small_cache.get("m", "text1", False)

        # Add new entry, should evict text2 (least recently used)
        await small_cache.set("m", "text4", False, "value4")

        assert small_cache.stats["memory_size"] == 3
        assert await small_cache.get("m", "text1", False) == "value1"
        assert await small_cache.get("m", "text2", False) is None  # Evicted
        assert await small_cache.get("m", "text3", False) == "value3"
        assert await small_cache.get("m", "text4", False) == "value4"

    @pytest.mark.asyncio
    async def test_cache_stats(self, cache):
        """Test cache statistics."""
        # Initial stats
        stats = cache.stats
        assert stats["memory_size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        # Add entries and access
        await cache.set("m", "text1", False, "v1")
        await cache.get("m", "text1", False)  # Hit
        await cache.get("m", "text2", False)  # Miss

        stats = cache.stats
        assert stats["memory_size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate_percent"] == 50.0

    @pytest.mark.asyncio
    async def test_cache_clear(self, cache):
        """Test clearing the cache."""
        await cache.set("m", "text1", False, "v1")
        await cache.set("m", "text2", False, "v2")

        assert cache.stats["memory_size"] == 2

        count = await cache.clear()

        assert count == 2
        assert cache.stats["memory_size"] == 0
        assert await cache.get("m", "text1", False) is None

    @pytest.mark.asyncio
    async def test_cache_cleanup_expired(self):
        """Test cleanup of expired entries."""
        short_ttl_cache = LRUCache(max_size=10, ttl_seconds=1)

        await short_ttl_cache.set("m", "text1", False, "v1")
        await short_ttl_cache.set("m", "text2", False, "v2")

        await asyncio.sleep(1.5)

        # Add a new entry that won't be expired
        await short_ttl_cache.set("m", "text3", False, "v3")

        # Cleanup should remove expired entries
        removed = await short_ttl_cache.cleanup_expired()
        assert removed == 2
        assert short_ttl_cache.stats["memory_size"] == 1

    @pytest.mark.asyncio
    async def test_concurrent_access(self, cache):
        """Test concurrent cache access is thread-safe."""
        async def write_task(i):
            await cache.set("m", f"text{i}", False, f"value{i}")

        async def read_task(i):
            return await cache.get("m", f"text{i}", False)

        # Concurrent writes
        await asyncio.gather(*[write_task(i) for i in range(50)])

        # Concurrent reads
        results = await asyncio.gather(*[read_task(i) for i in range(50)])

        # All writes should have succeeded
        for i, result in enumerate(results):
            assert result == f"value{i}"


class TestCacheEntry:
    """Test suite for CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(value="test", expires_at=time.time() + 3600, model="test_model")
        assert entry.value == "test"
        assert entry.model == "test_model"
        assert entry.hits == 0

    def test_cache_entry_hits(self):
        """Test incrementing hits."""
        entry = CacheEntry(value="test", expires_at=time.time() + 3600, model="test_model")
        entry.hits += 1
        assert entry.hits == 1
