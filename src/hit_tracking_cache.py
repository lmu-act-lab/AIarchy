"""
Hit-tracking cache implementation for evicting least-used entries.

This cache tracks access hits and evicts entries with minimum hit counts,
preserving frequently-accessed entries which is beneficial for Monte Carlo agents
that make similar queries.
"""


class HitTrackingCache:
    """
    Cache wrapper that tracks hit counts and evicts least-used entries.
    Preserves entries that are frequently accessed (beneficial for Monte Carlo agents).
    
    Attributes
    ----------
    max_size : int
        Maximum number of entries to keep in the cache.
    """
    def __init__(self, max_size: int = 5000):
        """
        Initialize the hit-tracking cache.
        
        Parameters
        ----------
        max_size : int, optional
            Maximum number of entries to keep. Defaults to 5000.
        """
        self._cache: dict = {}
        self._hit_counts: dict = {}  # Track hits per key
        self._access_order: list = []  # Track order for LRU fallback
        self.max_size = max_size
    
    def get(self, key):
        """
        Get value from cache and increment hit count.
        
        Parameters
        ----------
        key
            Cache key to retrieve.
            
        Returns
        -------
        Any
            Cached value if found, None otherwise.
        """
        if key in self._cache:
            self._hit_counts[key] = self._hit_counts.get(key, 0) + 1
            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def set(self, key, value):
        """
        Set value in cache, evicting least-hit entry if cache is full.
        
        Parameters
        ----------
        key
            Cache key to store.
        value
            Value to cache.
        """
        # If cache is full, evict least-hit entry
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_least_hit()
        
        self._cache[key] = value
        self._hit_counts[key] = self._hit_counts.get(key, 0)
        if key not in self._access_order:
            self._access_order.append(key)
    
    def _evict_least_hit(self):
        """Evict the entry with the minimum hit count."""
        if not self._hit_counts:
            # Fallback: use LRU if no hit data
            if self._access_order:
                oldest = self._access_order[0]
                del self._cache[oldest]
                if oldest in self._hit_counts:
                    del self._hit_counts[oldest]
                self._access_order.remove(oldest)
                return
        
        min_hits = min(self._hit_counts.values())
        candidates = [k for k, v in self._hit_counts.items() if v == min_hits]
        
        # If multiple with same hit count, use LRU (oldest among candidates)
        evict_key = None
        if len(candidates) > 1:
            # Find oldest among candidates
            for key in self._access_order:
                if key in candidates:
                    evict_key = key
                    break
        else:
            evict_key = candidates[0]
        
        if evict_key:
            del self._cache[evict_key]
            del self._hit_counts[evict_key]
            self._access_order.remove(evict_key)
    
    def __contains__(self, key):
        """
        Check if key is in cache.
        
        Parameters
        ----------
        key
            Cache key to check.
            
        Returns
        -------
        bool
            True if key exists in cache, False otherwise.
        """
        return key in self._cache
    
    def __len__(self):
        """
        Get cache size.
        
        Returns
        -------
        int
            Number of entries in cache.
        """
        return len(self._cache)
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._hit_counts.clear()
        self._access_order.clear()
    
    def get_stats(self):
        """
        Get cache statistics for debugging.
        
        Returns
        -------
        dict
            Dictionary with cache statistics including size, avg_hits, max_hits, min_hits.
        """
        if not self._hit_counts:
            return {"size": len(self._cache), "avg_hits": 0, "max_hits": 0, "min_hits": 0}
        hits = list(self._hit_counts.values())
        return {
            "size": len(self._cache),
            "avg_hits": sum(hits) / len(hits) if hits else 0,
            "max_hits": max(hits) if hits else 0,
            "min_hits": min(hits) if hits else 0,
        }

