#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Cache for fast recomputation
"""
from __future__ import division, print_function

import hashlib
import numpy as np

class Cache(object):
    _cache = {}
    def __init__(self, func, key="none"):
        self.key = key
        self.func = func
    def __call__(self, *args, **kwargs):
        hash = hashlib.md5()
        
        # hash arguments
        for arg in args:
            if isinstance(arg, np.ndarray):
                hash.update(arg.view(np.uint8))
            else:
                hash.update(arg)
        
        # hash keyword arguments
        kwds = list(kwargs.keys())
        kwds.sort()
        for k in kwds:
            hash.update(k)
            arg = kwargs[k]
            if isinstance(arg, np.ndarray):
                hash.update(arg.view(np.uint8))
            else:
                hash.update(arg)            
        
        hash.update(self.key)
        
        ref = hash.hexdigest()

        if ref in Cache._cache:
            return Cache._cache[ref]
        else:
            data = self.func(*args, **kwargs)
            Cache._cache[ref] = data
            return data
            