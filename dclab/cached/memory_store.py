import threading
import time


class MemoryStore:
    def __init__(self):
        """A dictionary-based in-memory store

        The store is thread-safe, because it is accessed by the
        threads that compute cached data and by the
        :class:`.StoreKeeper` thread at the same time.
        """
        self.data = {}
        self.lock = threading.Lock()

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        with self.lock:
            self.data[key][1] = time.monotonic()
            return self.data[key][0]

    def __len__(self):
        return len(self.data)

    def __setitem__(self, key, value):
        with self.lock:
            self.data[key] = [value, time.monotonic()]

    def clear(self):
        """Clear the memory store"""
        with self.lock:
            self.data.clear()

    def items(self):
        """Return all key-value pairs in the memory store"""
        with self.lock:
            return [(key, value) for key, (value, _) in self.data.items()]

    def pop(self, key):
        """Remove and return a value from the memory store"""
        with self.lock:
            return self.data.pop(key)[0]

    def remove_least_used_keys(self, n=1):
        """Remove n least-used keys from the memory store"""
        with self.lock:
            # Sort memory cache data by access time
            access = sorted([(ta, key)
                             for (key, (_, ta)) in self.data.items()])
            # Remove least-recently used data
            for _, key in access[:n]:
                self.data.pop(key)
