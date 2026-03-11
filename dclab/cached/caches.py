import atexit
import functools
import hashlib
import numbers
from typing import Any, Callable

import numpy as np

from .store_keeper import StoreKeeper


class umbrella_cache:
    _store_keeper = StoreKeeper.get_instance()
    _memory_store = _store_keeper.memory_store
    _disk_store = _store_keeper.disk_store
    _store_keeper_started = False
    atexit.register(_store_keeper.close)

    def __init__(self,
                 topic: str = "general",
                 bypass_memory_store: bool = False,
                 custom_handlers: dict[Any, Callable] = None,
                 ):
        """
                A cache that can be used to decorate methods that accept
        numpy ndarrays as arguments.

        - cache is based on dictionary
        - md5 hashes of method arguments are the dictionary keys
        - applicable decorator for all methods in a module
        - applicable to methods with the same name in different
          source files
        - set cache size with `cached.MAX_SIZE`
        - only one global cache is generated, there are no instances
          of `Cache`

        Notes
        -----
        If you are using other decorators with this decorator, please
        make sure to apply the `Cache` first (first line before method
        definition). This wrapper uses name, doc, and filename of the
        method to identify it. If another wrapper does not implement
        a unique `__doc__` and is applied to multiple methods, then
        `Cached` might return values of the wrong method.
        """
        # topic must be a valid directory name
        topic = "".join(
            [t for t in topic if t in "abcdefghijklmnopqrstuvwxyz-0123456789"])
        self.topic = topic or "general"
        self.use_memory_store = not bypass_memory_store
        self.custom_handlers = custom_handlers

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.use_memory_store and not self._disk_store:
                # shortcut when nothing is cached
                return func(*args, **kwargs)

            # Make sure the StoreKeeper thread is running
            if not umbrella_cache._store_keeper_started:
                if not umbrella_cache._store_keeper.is_alive():
                    umbrella_cache._store_keeper.start()
                umbrella_cache._store_keeper_started = True

            ref = compute_hash_for_cache(func, args, kwargs,
                                         custom_handlers=self.custom_handlers)
            key = f"{self.topic}/{ref[:3]}/{ref[3:6]}/{ref[6:]}"

            if self.use_memory_store and key in self._memory_store:
                return self._memory_store[key]
            elif key in self._disk_store:
                value = self._disk_store[key]
                if self.use_memory_store:
                    self._memory_store[key] = value
                return value
            else:
                value = func(*args, **kwargs)
                if self.use_memory_store:
                    self._memory_store[key] = value
                elif self._disk_store:
                    # Only write to DiskStore directly, if the memory store
                    # is disabled. Normally, the StoreKeeper does this in the
                    # background.
                    self._disk_store[key] = value
                return value

        return wrapper


def compute_hash_for_cache(func: Callable,
                           args: list,
                           kwargs: dict,
                           custom_handlers: dict[Any, Callable] = None):
    """Compute the hash for caching the function return value"""
    the_hash = hashlib.md5()

    # hash arguments
    update_hash(the_hash, args, custom_handlers=custom_handlers)

    # hash keyword arguments
    update_hash(the_hash, kwargs, custom_handlers=custom_handlers)

    # metadata
    update_hash(the_hash, func.__name__)
    update_hash(the_hash, func.__doc__)
    update_hash(the_hash, func.__code__.co_filename)

    return the_hash.hexdigest()


def update_hash(the_hash,
                arg,
                custom_handlers: dict[Any, Callable] = None
                ):
    """Update a hashing object with a Python object

    The argument can be a numpy array, a string, or a list/tuple
    of objects that are convertable to strings.
    """
    if isinstance(arg, numbers.Number):
        the_hash.update(f"{arg}".encode('utf-8'))
    elif isinstance(arg, str):
        the_hash.update(arg.encode('utf-8'))
    elif arg is None:
        the_hash.update(b"none")
    elif isinstance(arg, bytes):
        the_hash.update(arg)
    elif isinstance(arg, np.ndarray):
        the_hash.update(arg.tobytes())
    elif isinstance(arg, (list, tuple)):
        for a in arg:
            update_hash(the_hash, a,
                        custom_handlers=custom_handlers)
    elif isinstance(arg, dict):
        update_hash(the_hash, sorted(arg.items()),
                    custom_handlers=custom_handlers)
    else:
        if custom_handlers:
            for cls, handler in custom_handlers.items():
                if isinstance(cls, str) and arg.__class__.__name__ == cls:
                    # Handler identifier is the class name of the argument
                    update_hash(the_hash, handler(arg))
                    return  # no further checks necessary
                elif isinstance(arg, cls):
                    # Handler identifier is the class of the argument
                    update_hash(the_hash, handler(arg))
                    return  # no further checks necessary

        # Final option are classes that define `__getstate__`
        if hasattr(arg, '__getstate__'):
            try:
                update_hash(the_hash, arg.__getstate__())
            except BaseException:
                pass
            else:
                return  # no further checks necessary

        raise ValueError(f"No rule to hash object of type {type(arg)}")
