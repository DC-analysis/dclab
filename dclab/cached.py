#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Cache for fast "recomputation"
"""
from __future__ import division, print_function, unicode_literals

import gc
import hashlib
import numpy as np


MAX_SIZE = 100


class Cache(object):
    _cache = {}
    _keys = []

    def __init__(self, func):
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
        self.func = func

    def __call__(self, *args, **kwargs):
        self.ahash = hashlib.md5()

        # hash arguments
        for arg in args:
            self._update_hash(arg)

        # hash keyword arguments
        kwds = list(kwargs.keys())
        kwds.sort()
        for k in kwds:
            self._update_hash(k)
            self._update_hash(kwargs[k])

        # make sure we are caching for the correct method
        self._update_hash(self.func.__name__)
        self._update_hash(self.func.__doc__)
        self._update_hash(self.func.__code__.co_filename)

        ref = self.ahash.hexdigest()

        if ref in Cache._cache:
            return Cache._cache[ref]
        else:
            data = self.func(*args, **kwargs)
            Cache._cache[ref] = data
            Cache._keys.append(ref)
            if len(Cache._keys) > MAX_SIZE:
                delref = Cache._keys.pop(0)
                Cache._cache.pop(delref)
            return data

    @property
    def __doc__(self):
        return self.func.__doc__

    def _update_hash(self, arg):
        """ Takes an argument and updates the hash.
        The argument can be an np.array, string, or list
        of things that are convertable to strings.
        """
        if isinstance(arg, np.ndarray):
            self.ahash.update(arg.view(np.uint8))
        elif isinstance(arg, list):
            [self._update_hash(a) for a in arg]
        else:
            self.ahash.update(str(arg).encode('utf-8'))

    @staticmethod
    def clear_cache():
        """Remove all cached objects"""
        del Cache._keys
        for k in list(Cache._cache.keys()):
            it = Cache._cache.pop(k)
            del it
        del Cache._cache
        Cache._keys = []
        Cache._cache = {}
        gc.collect()
