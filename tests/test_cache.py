
import time
import sys

import pytest

from dclab import cached


@pytest.mark.skipif(sys.version_info < (3, 3),
                    reason="perf_counter requires python3.3 or higher")
def test_cache():
    "Test if caching happens"
    wait = .05

    @cached.Cache
    def func1(x):
        time.sleep(wait)
        return 2 * x

    a = func1(4)
    assert a == 8
    t1 = time.perf_counter()
    b = func1(4)
    t2 = time.perf_counter()
    assert t2 - t1 < wait
    assert b == a


@pytest.mark.skipif(sys.version_info < (3, 3),
                    reason="perf_counter requires python3.3 or higher")
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

    t1 = time.perf_counter()
    for ii in range(ss):
        func1(ii)
    t2 = time.perf_counter()
    assert t2 - t1 < wait

    func1(3.14)
    t3 = time.perf_counter()
    assert t3 - t2 > wait


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
