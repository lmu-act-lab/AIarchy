"""
Simple LRU cache implementation for efficient memory management.

This cache uses LRU (Least Recently Used) eviction, which is optimal for
Monte Carlo agents where consecutive agents make similar queries and recent
entries are most likely to be reused.
"""

from collections import OrderedDict


class LRUCache:
    """Simple LRU cache with hit tracking for statistics."""
    def __init__(self, max_size: int = 5000):
        self._cache: dict = {}
        self._access_order: OrderedDict = OrderedDict()
        self._hit_counts: dict = {}  # For statistics
        self.max_size = max_size
    
    def get(self, key):
        """Get value from cache, moving to end (most recently used)."""
        if key in self._cache:
            self._hit_counts[key] = self._hit_counts.get(key, 0) + 1
            self._access_order.move_to_end(key)
            return self._cache[key]
        return None
    
    def set(self, key, value):
        """Set value in cache, evicting oldest entry if full."""
        if len(self._cache) >= self.max_size and key not in self._cache:
            # Evict oldest (LRU)
            oldest, _ = self._access_order.popitem(last=False)
            del self._cache[oldest]
            if oldest in self._hit_counts:
                del self._hit_counts[oldest]
        
        self._cache[key] = value
        self._hit_counts[key] = self._hit_counts.get(key, 0)
        # Add to end (most recently used) - O(1) with OrderedDict
        if key not in self._access_order:
            self._access_order[key] = None
        else:
            self._access_order.move_to_end(key)
    
    def __contains__(self, key):
        return key in self._cache
    
    def __len__(self):
        return len(self._cache)
    
    def clear(self):
        self._cache.clear()
        self._access_order.clear()
        self._hit_counts.clear()
    
    def get_stats(self):
        """Get cache statistics."""
        if not self._hit_counts:
            return {"size": len(self._cache), "avg_hits": 0, "max_hits": 0, "min_hits": 0}
        hits = list(self._hit_counts.values())
        return {
            "size": len(self._cache),
            "avg_hits": sum(hits) / len(hits) if hits else 0,
            "max_hits": max(hits) if hits else 0,
            "min_hits": min(hits) if hits else 0,
        }

