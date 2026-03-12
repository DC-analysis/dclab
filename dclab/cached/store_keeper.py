import logging
import threading
import traceback

from .disk_store import DiskStore
from .memory_store import MemoryStore


class StoreKeeper(threading.Thread):
    _init_lock = threading.Lock()
    _instance = None

    def __init__(self, *args, **kwargs):
        """Background thread for managing in-memory and persistent caches

        The ``StoreKeeper`` class allows you to modify the caching
        behavior of all methods decorated with :class:`.umbrella_cache`.

        To access the global instance, use the
        :func:`.StoreKeeper.get_instance` method.
        """
        super(StoreKeeper, self).__init__(daemon=True,
                                          name="StoreKeeper",
                                          *args, **kwargs)
        self.event_exit = threading.Event()
        self.logger = logging.getLogger(__name__)

        #: global volatile memory store
        self.memory_store = MemoryStore()
        #: global persistent disk store
        self.disk_store = DiskStore()
        #: housekeeping interval [s]
        self.interval = 1
        #: maximum number of keys in the memory store
        self.memory_store_size = 500
        #: maximum size of the disk store
        self.disk_store_size_bytes = 1024**3

    @classmethod
    def get_instance(cls):
        """Return the global ``StoreKeeper`` instance"""
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = StoreKeeper()
        return cls._instance

    def clear(self):
        """Clear memory and disk store"""
        self.memory_store.clear()
        self.disk_store.clear()

    def close(self):
        """Stop the background thread"""
        self.event_exit.set()
        if self.is_alive():
            self.join()

    def perform_tasks(self):
        """Perform memory and disk management tasks

        This method is called from within the ``run`` method and
        should not be called manually.
        """
        memory_store = self.memory_store
        disk_store = self.disk_store
        # Move data from memory store to disk store
        if disk_store:
            num_stored = 0
            for key, value in memory_store.items():
                try:
                    if key not in disk_store:
                        disk_store[key] = value
                        num_stored += 1
                except BaseException:
                    self.logger.error(traceback.format_exc())
            self.logger.info(
                f"Added {num_stored} entries to persistent cache")

            # Remove data from disk store
            try:
                disk_store.remove_old_files(
                    max_bytes=self.disk_store_size_bytes)
            except BaseException:
                self.logger.error(traceback.format_exc())

            # Remove stale locks
            try:
                disk_store.remove_stale_locks()
            except BaseException:
                self.logger.error(traceback.format_exc())

        # Honor `memory_store_size`
        to_remove = len(memory_store) - self.memory_store_size
        if to_remove > 0:
            try:
                memory_store.remove_least_used_keys(to_remove)
            except BaseException:
                self.logger.error(traceback.format_exc())
            self.logger.info(
                f"Removed {to_remove} entries from volatile cache")

    def run(self):
        """Enter the main loop"""
        self.logger.info("Caching StoreKeeper thread started")
        disk_store = self.disk_store
        if disk_store:
            # update disk store index
            disk_store.remove_old_files(max_bytes=self.disk_store_size_bytes)
        while not self.event_exit.wait(self.interval):
            self.perform_tasks()

    def set_interval(self, interval):
        """Set the interval in seconds at which ``perform_tasks`` is called"""
        self.interval = interval

    def set_memory_store_size(self, memory_store_size):
        """Set the allowed number of values in the in-memory cache

        Since the memory store is tidied up in the background at fixed
        intervals, the store size might temporarily exceed the set value.
        """
        self.memory_store_size = memory_store_size

    def set_disk_store_path(self, path):
        """Set the path where to store persistent cache data

        If this method is not called, then the disk store is disabled.
        """
        self.disk_store.set_path(path)

    def set_disk_store_size_bytes(self, disk_store_size_bytes):
        """Set maximum size of the disk store in bytes

        This number limits the size the disk store is allowed to
        occupy on disk. Due to metadata overhead, the actual size
        is slightly larger.

        Since the disk store is tidied up in the background at fixed
        intervals, the store size might temporarily exceed the set value.
        """
        self.disk_store_size_bytes = disk_store_size_bytes
