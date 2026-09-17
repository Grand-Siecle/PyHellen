"""
Tests for the cache module.
"""

import asyncio
import pytest
import time

from app.core.cache import LRUCache, CacheEntry, HybridCache


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
        await cache.set("model1", "Text", False, {"lower": False})
        await cache.set("model1", "Text", True, {"lower": True})

        result1 = await cache.get("model1", "Text", False)
        result2 = await cache.get("model1", "Text", True)

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

    def test_cache_entry_expiration(self):
        """Test cache entry expiration detection."""
        # Create entry that expires in the past
        expired_entry = CacheEntry(
            value="expired",
            expires_at=time.time() - 100,
            model="test_model"
        )
        assert time.time() > expired_entry.expires_at

        # Create entry that expires in the future
        valid_entry = CacheEntry(
            value="valid",
            expires_at=time.time() + 3600,
            model="test_model"
        )
        assert time.time() < valid_entry.expires_at


class TestCacheClearModel:
    """Test suite for model-specific cache clearing."""

    @pytest.mark.asyncio
    async def test_clear_model_entries(self, cache):
        """Test clearing cache entries for a specific model."""
        # Add entries for different models
        await cache.set("model_a", "text1", False, "value1")
        await cache.set("model_a", "text2", False, "value2")
        await cache.set("model_b", "text1", False, "value3")

        assert cache.stats["memory_size"] == 3

        # Clear only model_a entries
        cleared = await cache.clear_model("model_a")

        assert cleared == 2
        assert cache.stats["memory_size"] == 1
        assert await cache.get("model_a", "text1", False) is None
        assert await cache.get("model_a", "text2", False) is None
        assert await cache.get("model_b", "text1", False) == "value3"

    @pytest.mark.asyncio
    async def test_clear_model_no_entries(self, cache):
        """Test clearing model with no entries returns 0."""
        await cache.set("model_a", "text1", False, "value1")
        cleared = await cache.clear_model("nonexistent_model")
        assert cleared == 0
        assert cache.stats["memory_size"] == 1


class TestCacheKeyGeneration:
    """Test suite for cache key generation."""

    @pytest.mark.asyncio
    async def test_key_varies_with_model(self, cache):
        """Test that different models produce different cache keys."""
        await cache.set("model1", "same_text", False, "result1")
        await cache.set("model2", "same_text", False, "result2")

        assert await cache.get("model1", "same_text", False) == "result1"
        assert await cache.get("model2", "same_text", False) == "result2"

    @pytest.mark.asyncio
    async def test_key_varies_with_lower_flag(self, cache):
        """Test that lower flag affects cache key."""
        await cache.set("model", "TEXT", False, "uppercase_result")
        await cache.set("model", "TEXT", True, "lowercase_result")

        assert await cache.get("model", "TEXT", False) == "uppercase_result"
        assert await cache.get("model", "TEXT", True) == "lowercase_result"

    @pytest.mark.asyncio
    async def test_key_with_special_characters(self, cache):
        """Test cache with special characters in text."""
        special_text = "Test with émojis 🎉 and ümlauts äöü"
        await cache.set("model", special_text, False, "special_result")
        result = await cache.get("model", special_text, False)
        assert result == "special_result"

    @pytest.mark.asyncio
    async def test_key_with_long_text(self, cache):
        """Test cache with very long text."""
        long_text = "Lorem ipsum " * 1000  # ~12000 characters
        await cache.set("model", long_text, False, "long_result")
        result = await cache.get("model", long_text, False)
        assert result == "long_result"


class TestCacheStatsEdgeCases:
    """Test suite for cache statistics edge cases."""

    @pytest.mark.asyncio
    async def test_stats_with_zero_requests(self, cache):
        """Test hit rate is 0 when no requests made."""
        stats = cache.stats
        assert stats["hit_rate_percent"] == 0

    @pytest.mark.asyncio
    async def test_stats_all_misses(self, cache):
        """Test hit rate is 0 when all requests are misses."""
        await cache.get("model", "text1", False)
        await cache.get("model", "text2", False)

        stats = cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 2
        assert stats["hit_rate_percent"] == 0

    @pytest.mark.asyncio
    async def test_stats_all_hits(self, cache):
        """Test hit rate is 100 when all requests are hits."""
        await cache.set("model", "text", False, "value")
        await cache.get("model", "text", False)
        await cache.get("model", "text", False)

        stats = cache.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 0
        assert stats["hit_rate_percent"] == 100.0

    @pytest.mark.asyncio
    async def test_stats_persistence_flag(self, cache):
        """Test that persistence flag is correctly reported in stats."""
        stats = cache.stats
        assert "persistence_enabled" in stats
        # The test fixture disables persistence
        assert stats["persistence_enabled"] is False


class TestCacheRaceConditions:
    """Test suite for concurrent edge cases."""

    @pytest.mark.asyncio
    async def test_concurrent_set_same_key(self, cache):
        """Test concurrent sets to the same key."""
        async def set_task(value):
            await cache.set("model", "text", False, value)
            return value

        # Run multiple concurrent sets
        results = await asyncio.gather(*[set_task(f"value{i}") for i in range(10)])

        # Should have exactly one entry
        assert cache.stats["memory_size"] == 1

        # Should have one of the values
        final_value = await cache.get("model", "text", False)
        assert final_value in results

    @pytest.mark.asyncio
    async def test_concurrent_get_and_set(self, cache):
        """Test concurrent gets and sets."""
        await cache.set("model", "text", False, "initial")

        async def get_task():
            return await cache.get("model", "text", False)

        async def set_task(value):
            await cache.set("model", "text", False, value)

        # Mix gets and sets
        tasks = [get_task() for _ in range(5)] + [set_task("updated")]
        results = await asyncio.gather(*tasks)

        # Gets should return either initial or updated
        get_results = [r for r in results if r is not None]
        assert all(r in ["initial", "updated"] for r in get_results)


class TestCacheKeyVersioning:
    """Results must never be served across model/library versions or inference settings."""

    @pytest.mark.asyncio
    async def test_fingerprint_change_invalidates_entries(self):
        fingerprint = {"value": "pie=0.1.5|quantize=False"}
        cache = HybridCache(persist=False, fingerprint=lambda model: fingerprint["value"])
        await cache.set("freem", "Et le Roy", False, "float result")

        fingerprint["value"] = "pie=0.1.5|quantize=True"

        assert await cache.get("freem", "Et le Roy", False) is None

    @pytest.mark.asyncio
    async def test_fingerprint_receives_model_name(self):
        seen = []
        cache = HybridCache(persist=False, fingerprint=lambda model: seen.append(model) or "v1")
        await cache.set("lasla", "Gallia", False, "r")
        await cache.get("lasla", "Gallia", False)
        assert set(seen) == {"lasla"}

    @pytest.mark.asyncio
    async def test_lower_flag_shares_entry_with_equivalent_text(self, cache):
        """process_text lowercases before tagging, so ("ABC", lower=True) and ("abc", lower=False) are the same input."""
        await cache.set("m", "ABC", True, "tagged abc")

        assert await cache.get("m", "abc", False) == "tagged abc"
        assert await cache.get("m", "ABC", False) is None


class TestCacheLimits:
    """Configurable switches and size limits."""

    @pytest.mark.asyncio
    async def test_disabled_cache_stores_nothing(self):
        cache = HybridCache(persist=False, enabled=False)
        await cache.set("m", "text", False, "value")
        assert await cache.get("m", "text", False) is None
        assert cache.stats["memory_size"] == 0

    @pytest.mark.asyncio
    async def test_entry_larger_than_max_entry_bytes_is_not_cached(self):
        cache = HybridCache(persist=False, max_entry_bytes=100)
        await cache.set("m", "small", False, ["x"])
        await cache.set("m", "big", False, ["x" * 500])

        assert await cache.get("m", "small", False) == ["x"]
        assert await cache.get("m", "big", False) is None

    @pytest.mark.asyncio
    async def test_memory_byte_budget_evicts_least_recently_used(self):
        cache = HybridCache(max_size=1000, persist=False, max_bytes=250)
        for name in ("a", "b", "c"):
            await cache.set("m", name, False, [name * 90])  # ~96 bytes of JSON each
        # 3 entries (~288 bytes) exceed 250 bytes: "a" (least recently used) must be gone
        assert await cache.get("m", "a", False) is None
        assert await cache.get("m", "b", False) == ["b" * 90]
        assert await cache.get("m", "c", False) == ["c" * 90]
        assert cache.stats["memory_bytes"] <= 250


class TestCacheBulkOperations:
    """get_many / set_many serve batch endpoints with one database round-trip."""

    @pytest.mark.asyncio
    async def test_get_many_preserves_order_and_counts(self, cache):
        await cache.set("m", "t1", False, "v1")
        await cache.set("m", "t3", False, "v3")

        assert await cache.get_many("m", ["t1", "t2", "t3"], False) == ["v1", None, "v3"]
        assert cache.stats["hits"] == 2
        assert cache.stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_set_many_then_get_many(self, cache):
        await cache.set_many("m", [("t1", "v1"), ("t2", "v2")], False)
        assert await cache.get_many("m", ["t2", "t1"], False) == ["v2", "v1"]

    @pytest.mark.asyncio
    async def test_bulk_operations_persist_to_database(self):
        cache = HybridCache(max_size=10, ttl_seconds=60, persist=True)
        await cache.clear()
        await cache.set_many("lasla", [("Gallia est", ["g"]), ("omnis divisa", ["o"])], False)

        restarted = HybridCache(max_size=10, ttl_seconds=60, persist=True)
        assert await restarted.get_many("lasla", ["omnis divisa", "Gallia est", "absent"], False) == [
            ["o"],
            ["g"],
            None,
        ]


class TestCacheGetOrCompute:
    """Concurrent identical requests must share one computation."""

    @pytest.mark.asyncio
    async def test_concurrent_misses_compute_once(self, cache):
        calls = 0

        async def compute():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.05)
            return ["tagged"]

        outcomes = await asyncio.gather(*(cache.get_or_compute("m", "same text", False, compute) for _ in range(5)))

        assert calls == 1
        assert [value for value, _ in outcomes] == [["tagged"]] * 5
        assert sorted(from_cache for _, from_cache in outcomes) == [False, True, True, True, True]
        assert await cache.get("m", "same text", False) == ["tagged"]

    @pytest.mark.asyncio
    async def test_compute_error_propagates_and_is_not_cached(self, cache):
        async def failing():
            await asyncio.sleep(0.01)
            raise RuntimeError("CUDA error")

        outcomes = await asyncio.gather(
            *(cache.get_or_compute("m", "text", False, failing) for _ in range(3)), return_exceptions=True
        )
        assert all(isinstance(outcome, RuntimeError) for outcome in outcomes)

        async def working():
            return "ok"

        assert await cache.get_or_compute("m", "text", False, working) == ("ok", False)

    @pytest.mark.asyncio
    async def test_disabled_cache_always_computes(self):
        cache = HybridCache(persist=False, enabled=False)
        calls = 0

        async def compute():
            nonlocal calls
            calls += 1
            return "v"

        await cache.get_or_compute("m", "t", False, compute)
        await cache.get_or_compute("m", "t", False, compute)
        assert calls == 2


class TestCachePersistence:
    """SQLite access must not block the event loop and must stay consistent with memory."""

    @pytest.mark.asyncio
    async def test_database_calls_run_outside_event_loop_thread(self):
        import threading

        loop_thread = threading.get_ident()
        cache = HybridCache(max_size=10, ttl_seconds=60, persist=True)
        repo = cache._get_repo()
        threads = []
        for name in ("get_many", "set_many"):
            original = getattr(repo, name)

            def spy(*args, _original=original, **kwargs):
                threads.append(threading.get_ident())
                return _original(*args, **kwargs)

            setattr(repo, name, spy)

        await cache.set("lasla", "Gallia", False, ["g"])
        cache._cache.clear()
        await cache.get("lasla", "Gallia", False)

        assert len(threads) == 2
        assert loop_thread not in threads

    @pytest.mark.asyncio
    async def test_promoted_entry_keeps_database_expiry(self):
        cache = HybridCache(max_size=10, ttl_seconds=60, persist=True)
        await cache.clear()
        await cache.set("lasla", "Gallia", False, ["g"])
        stored_expiry = cache._cache[cache._generate_key("lasla", "Gallia", False)].expires_at

        await asyncio.sleep(0.05)
        restarted = HybridCache(max_size=10, ttl_seconds=60, persist=True)
        await restarted.get("lasla", "Gallia", False)
        promoted_expiry = restarted._cache[restarted._generate_key("lasla", "Gallia", False)].expires_at

        assert promoted_expiry == pytest.approx(stored_expiry, abs=1.0)
        assert promoted_expiry < stored_expiry + 0.04  # not reset to "now + ttl"

    @pytest.mark.asyncio
    async def test_clear_counts_distinct_entries(self):
        cache = HybridCache(max_size=10, ttl_seconds=60, persist=True)
        await cache.clear()
        await cache.set_many("lasla", [("a", 1), ("b", 2)], False)
        await cache.set("grc", "c", False, 3)

        assert await cache.clear_model("lasla") == 2
        assert await cache.clear() == 1

    @pytest.mark.asyncio
    async def test_get_stats_includes_database(self):
        cache = HybridCache(max_size=10, ttl_seconds=60, persist=True)
        await cache.clear()
        await cache.set("lasla", "Gallia", False, ["g"])

        stats = await cache.get_stats()

        assert stats["memory_size"] == 1
        assert stats["database"]["size"] == 1


class TestCacheCleanupTask:
    """Expired entries are purged periodically without an explicit API call."""

    @pytest.mark.asyncio
    async def test_periodic_cleanup_removes_expired_entries(self):
        cache = HybridCache(max_size=10, ttl_seconds=0.1, persist=False)
        await cache.set("m", "text", False, "v")

        task = cache.start_cleanup_task(interval_seconds=0.05)
        try:
            await asyncio.sleep(0.3)
            assert cache.stats["memory_size"] == 0
        finally:
            task.cancel()
