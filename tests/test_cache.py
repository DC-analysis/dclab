#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import time

from dclab import cached


def test_cache():
    "Test if caching happens"
    wait = .05

    @cached.Cache
    def func1(x):
        time.sleep(wait)
        return 2 * x

    a = func1(4)
    assert a == 8
    t1 = time.time()
    b = func1(4)
    t2 = time.time()
    assert t2 - t1 < wait
    assert b == a


def test_cache_size():
    "Create more caches than cached.MAX_SIZE"
    ss = 10
    wait = .01
    cached.MAX_SIZE = ss

    @cached.Cache
    def func1(x):
        time.sleep(wait)
        return 2 * x
    for ii in range(ss):
        func1(ii)

    t1 = time.time()
    for ii in range(ss):
        func1(ii)
    t2 = time.time()
    assert t2 - t1 < wait

    func1(3.14)
    t3 = time.time()
    assert t3 - t2 > wait


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
