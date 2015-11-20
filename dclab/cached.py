#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Cache for fast recomputation
"""
from __future__ import division, print_function

import hashlib
import numpy as np

class Cache(object):
    """
    A cache that can be used to decorate methods that accept
    numpy ndarrays as arguments.
    """
    _cache = {}
    def __init__(self, func):
        self.func = func
    def __call__(self, *args, **kwargs):
        ahash = hashlib.md5()

        # hash arguments
        for arg in args:
            if isinstance(arg, np.ndarray):
                ahash.update(arg.view(np.uint8))
            else:
                ahash.update(arg)
        
        # hash keyword arguments
        kwds = list(kwargs.keys())
        kwds.sort()
        for k in kwds:
            ahash.update(k)
            arg = kwargs[k]
            if isinstance(arg, np.ndarray):
                ahash.update(arg.view(np.uint8))
            else:
                ahash.update(arg)            
        
        # make sure we are caching for the correct method
        ahash.update(self.func.func_name)   
        ahash.update(self.func.func_code.co_filename)
        
        ref = ahash.hexdigest()

        if ref in Cache._cache:
            return Cache._cache[ref]
        else:
            data = self.func(*args, **kwargs)
            Cache._cache[ref] = data
            return data
            