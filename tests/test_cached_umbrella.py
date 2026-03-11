import hashlib
import time
import sys

import numpy as np

import pytest

from dclab import cached
from dclab.cached import caches


@pytest.fixture
def store_keeper():
    store_keeper = cached.umbrella_cache._store_keeper
    interval = store_keeper.interval
    store_keeper.interval = 0.5
    store_keeper.clear()
    yield store_keeper
    store_keeper.interval = interval


@pytest.mark.skipif(sys.version_info < (3, 3),
                    reason="perf_counter requires python3.3 or higher")
def test_umbrella_cache():
    """Test if caching works"""
    wait = .05

    @cached.umbrella_cache()
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
def test_umbrella_cache_disabled():
    """Test if caching works"""
    wait = .05

    @cached.umbrella_cache(bypass_memory_store=True)
    def func2(x):
        time.sleep(wait)
        return 2 * x

    a = func2(4)
    assert a == 8
    t1 = time.perf_counter()
    b = func2(4)
    t2 = time.perf_counter()
    assert t2 - t1 >= wait
    assert b == a


@pytest.mark.skipif(sys.version_info < (3, 3),
                    reason="perf_counter requires python3.3 or higher")
def test_umbrella_cache_disk_store(tmp_path, store_keeper):
    """Test if caching works"""
    store_keeper.set_disk_store_path(tmp_path)

    wait = .05

    @cached.umbrella_cache(bypass_memory_store=True)
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
def test_umbrella_cache_disk_store_hybrid(tmp_path, store_keeper):
    """Test if caching works"""
    store_keeper.set_disk_store_path(tmp_path)

    wait = .05

    @cached.umbrella_cache()
    def func1(x):
        time.sleep(wait)
        return 2 * x

    a = func1(4)
    assert a == 8

    # wait for the data to be written to the disk store
    time.sleep(1)

    # cause the wrapper to fetch the data from the disk store
    store_keeper.memory_store.clear()

    t1 = time.perf_counter()
    b = func1(4)
    t2 = time.perf_counter()
    assert t2 - t1 < wait
    assert b == a


@pytest.mark.skipif(sys.version_info < (3, 3),
                    reason="perf_counter requires python3.3 or higher")
def test_umbrella_cache_size(store_keeper):
    """Create more cache entries than memory_store_size"""
    ss = 10
    wait = .1

    store_keeper.set_memory_store_size(10)
    store_keeper.set_interval(0.2)

    @cached.umbrella_cache()
    def func3(x):
        time.sleep(wait)
        return 2 * x

    for ii in range(ss):
        func3(ii)
        assert len(store_keeper.memory_store) == ii + 1

    # Performance test (make sure caching works)
    t1 = time.perf_counter()
    for ii in range(ss):
        func3(ii)
        assert len(store_keeper.memory_store) == ss

    t2 = time.perf_counter()
    assert t2 - t1 < wait

    assert len(store_keeper.memory_store) == 10

    # Call with argument that hasn't been called before
    func3(3.14)
    t3 = time.perf_counter()
    # time.sleep is apparently not that accurate, hence the "/2".
    assert t3 - t2 > wait / 2

    # Wait for the store_keeper to remove the item
    time.sleep(1.1)

    assert len(store_keeper.memory_store) == 10


def test_update_hash(store_keeper):
    data = [
        "a string",
        b"some bytes",
        123,
        1.23,
        np.float64(1.234),
        np.int64(1234),
        np.arange(10),
    ]
    hasher = hashlib.md5()
    caches.update_hash(hasher, data)
    assert hasher.hexdigest() == "995204fa74d9f6e66ea055f9af0379b3"


def test_update_hash_setstate_classes(store_keeper):
    class CustomClass:
        def __getstate__(self):
            return np.arange(10)

    data = [
        "a string",
        b"some bytes",
        123,
        1.23,
        np.float64(1.234),
        np.int64(1234),
        CustomClass(),
    ]
    hasher = hashlib.md5()
    caches.update_hash(hasher, data)
    assert hasher.hexdigest() == "995204fa74d9f6e66ea055f9af0379b3"


def test_update_hash_custom_handlers(store_keeper):
    class HandledClass:
        def __init__(self):
            self.data = np.arange(10)

    def custom_handler(obj):
        return obj.data

    data = [
        "a string",
        b"some bytes",
        123,
        1.23,
        np.float64(1.234),
        np.int64(1234),
        HandledClass(),
    ]
    hasher = hashlib.md5()
    caches.update_hash(hasher, data,
                       custom_handlers={HandledClass: custom_handler})
    assert hasher.hexdigest() == "995204fa74d9f6e66ea055f9af0379b3"


def test_update_hash_custom_handlers_string(store_keeper):
    class HandledClass:
        def __init__(self):
            self.data = np.arange(10)

    def custom_handler(obj):
        return obj.data

    data = [
        "a string",
        b"some bytes",
        123,
        1.23,
        np.float64(1.234),
        np.int64(1234),
        HandledClass(),
    ]
    hasher = hashlib.md5()
    caches.update_hash(hasher, data,
                       custom_handlers={"HandledClass": custom_handler})
    assert hasher.hexdigest() == "995204fa74d9f6e66ea055f9af0379b3"


def test_update_hash_custom_handlers_fail(store_keeper):
    class HandledClass:
        def __init__(self):
            self.data = np.arange(10)

    class UnHandledClass:
        def __init__(self):
            self.data = np.arange(10)

        def __getstate__(self):
            raise ValueError("Not supported")

    def custom_handler(obj):
        return obj.data

    data = [
        "a string",
        b"some bytes",
        123,
        1.23,
        np.float64(1.234),
        np.int64(1234),
        HandledClass(),
        UnHandledClass(),
    ]
    hasher = hashlib.md5()

    with pytest.raises(ValueError, match="No rule to hash"):
        caches.update_hash(hasher, data,
                           custom_handlers={HandledClass: custom_handler})
