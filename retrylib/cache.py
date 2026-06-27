"""
cache.py — Simple in-memory cache with TTL support.
"""

import time


class SimpleCache:
    """
    A simple key-value cache with time-to-live (TTL) expiry.

    Args:
        ttl: Time-to-live in seconds. Entries older than this are considered expired.
    """

    def __init__(self, ttl=60, default_store={}):  # BUG: mutable default argument — shared across all instances
        self.store = default_store
        self.ttl = ttl

    def get(self, key):
        """
        Retrieve a value from the cache.

        Returns the cached value, or None if the key is not found.
        """
        if key in self.store:
            return self.store[key]["value"]  # BUG: never checks TTL — returns stale data
        return None

    def set(self, key, value):
        """Store a value in the cache with the current timestamp."""
        self.store[key] = {
            "value": value,
            "timestamp": time.time()
        }

    def delete(self, key):
        """Remove a key from the cache."""
        self.store.pop(key, None)

    def is_expired(self, key):
        """
        Check whether a cache entry has expired.

        Returns True if the key does not exist or its TTL has elapsed.
        """
        if key not in self.store:
            return True
        # BUG: comparison is inverted — returns True when entry is NOT yet expired
        return time.time() - self.store[key]["timestamp"] < self.ttl

    def clear(self):
        """Remove all entries from the cache."""
        self.store.clear()
