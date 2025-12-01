"""Tests for caching infrastructure.

Tests LRU cache and result cache implementations.
"""

import time

import numpy as np

from dta.dti.cache import LRUCache, ResultCache, get_result_cache


class TestLRUCache:
    """Tests for LRU cache."""

    def test_basic_operations(self) -> None:
        """Test basic set/get operations."""
        cache = LRUCache(max_size=3, default_ttl=10)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_eviction(self) -> None:
        """Test LRU eviction when cache is full."""
        cache = LRUCache(max_size=3, default_ttl=10)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Add one more (should evict key1)
        cache.set("key4", "value4")
        assert cache.get("key1") is None
        assert cache.get("key4") == "value4"

        assert cache.size == 3

    def test_ttl_expiration(self) -> None:
        """Test cache TTL expiration."""
        cache = LRUCache(max_size=10, default_ttl=1)

        cache.set("key", "value", ttl=1)
        assert cache.get("key") == "value"

        time.sleep(1.1)
        assert cache.get("key") is None

    def test_statistics(self) -> None:
        """Test cache statistics."""
        cache = LRUCache(max_size=10)

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestResultCache:
    """Tests for result cache."""

    def test_generate_key(self) -> None:
        """Test cache key generation."""
        cache = ResultCache(max_size=10)

        key = cache.generate_key("algorithm/ndvi", {"raster": "test.tif"})
        assert isinstance(key, str)
        assert len(key) == 16

    def test_cache_result(self) -> None:
        """Test caching results."""
        cache = ResultCache(max_size=10)

        result = {"ndvi": np.array([0.5, 0.6, 0.7])}
        cache.set("algorithm/ndvi", {"raster": "test.tif"}, result)

        cached = cache.get("algorithm/ndvi", {"raster": "test.tif"})
        assert cached is not None
        assert "ndvi" in cached

    def test_global_result_cache(self) -> None:
        """Test global result cache."""
        cache = get_result_cache()
        cache.clear()

        cache.set("test", {"input": 1}, {"output": 2})
        result = cache.get("test", {"input": 1})
        assert result == {"output": 2}
