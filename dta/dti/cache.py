"""Caching layer for DTA.

Provides in-memory caching with TTL support for expensive operations.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import threading
import time
from typing import Any


@dataclass
class CacheEntry:
    """Cache entry with TTL."""

    key: str
    value: Any
    created_at: float
    ttl: int  # seconds
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() - self.created_at > self.ttl


class LRUCache:
    """LRU (Least Recently Used) cache with TTL support."""

    def __init__(self, max_size: int = 100, default_ttl: int = 3600) -> None:
        """Initialize cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds (default: 1 hour)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            self._misses += 1
            return None

        entry = self._cache[key]

        # Check expiration
        if entry.is_expired:
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.hits += 1
        self._hits += 1

        return entry.value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (None = use default)
        """
        if ttl is None:
            ttl = self.default_ttl

        # Remove if exists
        if key in self._cache:
            del self._cache[key]

        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        # Add new entry
        entry = CacheEntry(key=key, value=value, created_at=time.time(), ttl=ttl)
        self._cache[key] = entry

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def remove(self, key: str) -> None:
        """Remove specific key from cache.

        Args:
            key: Cache key
        """
        if key in self._cache:
            del self._cache[key]

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired]

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "entries": [
                {
                    "key": key,
                    "hits": entry.hits,
                    "age_seconds": time.time() - entry.created_at,
                    "ttl": entry.ttl,
                }
                for key, entry in list(self._cache.items())[:10]  # First 10
            ],
        }


class ResultCache:
    """Cache for algorithm/model results."""

    def __init__(self, max_size: int = 50, default_ttl: int = 1800) -> None:
        """Initialize result cache.

        Args:
            max_size: Maximum cache entries
            default_ttl: Default TTL (30 minutes)
        """
        self._cache = LRUCache(max_size=max_size, default_ttl=default_ttl)

    def generate_key(self, component_id: str, inputs: dict[str, Any]) -> str:
        """Generate cache key from component and inputs.

        Args:
            component_id: Component identifier
            inputs: Input parameters

        Returns:
            Cache key (hash)
        """
        # Create deterministic key from inputs
        inputs_str = json.dumps(inputs, sort_keys=True)
        key_str = f"{component_id}:{inputs_str}"

        # Hash for compact key
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def get(self, component_id: str, inputs: dict[str, Any]) -> Any | None:
        """Get cached result.

        Args:
            component_id: Component identifier
            inputs: Input parameters

        Returns:
            Cached result or None
        """
        key = self.generate_key(component_id, inputs)
        return self._cache.get(key)

    def set(self, component_id: str, inputs: dict[str, Any], result: Any, ttl: int | None = None) -> None:
        """Cache result.

        Args:
            component_id: Component identifier
            inputs: Input parameters
            result: Result to cache
            ttl: Optional TTL override
        """
        key = self.generate_key(component_id, inputs)
        self._cache.set(key, result, ttl)

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()


# Global caches
_result_cache: ResultCache | None = None
_result_cache_lock = threading.Lock()


def get_result_cache() -> ResultCache:
    """Get global result cache.

    Returns:
        Result cache instance
    """
    global _result_cache
    if _result_cache is None:
        with _result_cache_lock:
            if _result_cache is None:
                _result_cache = ResultCache()
    return _result_cache
