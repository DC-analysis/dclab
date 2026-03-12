import json
import multiprocessing as mp
import os
import pathlib
import time

import numpy as np
from dclab import cached
from dclab.cached.disk_store import LockFile

import pytest


def lock_acquirer(path, ready_counter, start_bit,
                  done_counter, result_counter):
    with ready_counter.get_lock():
        ready_counter.value += 1

    # wait for the parent process to start
    while start_bit.value == 0:
        pass

    with LockFile(path) as lf:
        acquired = lf.acquired
        with done_counter.get_lock():
            done_counter.value -= 1
        while True:
            if done_counter.value == 0:
                break
            time.sleep(0.1)
    if acquired:
        with result_counter.get_lock():
            result_counter.value += 1


def test_lockfile(tmp_path):
    lock_path = tmp_path / "locked.lock"
    with LockFile(lock_path) as lf:
        assert lf.acquired

        with LockFile(lock_path) as lf2:
            assert not lf2.acquired

        assert lock_path.exists()

    assert not lock_path.exists()


def test_lockfile_multiprocessing(tmp_path):
    """Let multiple workers compete for locking a file"""
    num_workers = 7
    mp_spawn = mp.get_context("spawn")
    lock_path = tmp_path / "locked.lock"
    done_counter = mp_spawn.Value("i", num_workers)
    ready_counter = mp_spawn.Value("i", 0)
    start_bit = mp_spawn.Value("i", 0)
    result_counter = mp_spawn.Value("i", 0)
    workers = []
    for ii in range(num_workers):
        p = mp_spawn.Process(
            target=lock_acquirer,
            args=(lock_path, ready_counter, start_bit, done_counter,
                  result_counter))
        p.start()
        workers.append(p)

    # let the workers get ready
    while ready_counter.value != num_workers:
        time.sleep(0.1)

    # start the competition
    start_bit.value = 1

    # wait for the competition to finish
    for p in workers:
        p.join()

    assert result_counter.value == 1
    assert not lock_path.exists()


def test_disk_store_disabled():
    store = cached.DiskStore()

    assert not bool(store)
    with pytest.raises(RuntimeError, match="set a disk cache path"):
        store.assert_disk_store_path()


def test_disk_store_dunder(tmp_path):
    store = cached.DiskStore(tmp_path)
    test_dict = {"number": 2,
                 "number2": np.int64(2),
                 "list": [1, np.pi, 3.0],
                 "peter": "hans",
                 "pi": np.float64(np.pi)}
    store["hans/peter"] = test_dict

    assert bool(store)
    assert "hans/peter" in store
    assert "quamaklik" not in store

    store.clear()
    assert not tmp_path.exists()


def test_disk_store_fail(tmp_path):
    store = cached.DiskStore(tmp_path)

    with pytest.raises(NotImplementedError):
        store["invalid/data"] = np


def test_disk_store_json(tmp_path):
    store = cached.DiskStore(tmp_path)
    test_dict = {"number": 2,
                 "number2": np.int64(2),
                 "list": [1, np.pi, 3.0],
                 "peter": "hans",
                 "pi": np.float64(np.pi)}
    store["hans/peter"] = test_dict

    meta_path = tmp_path / "hans/peter_meta.json"
    assert meta_path.exists()

    data_path = tmp_path / "hans/peter.json"
    assert json.loads(data_path.read_text()) == test_dict

    assert store["hans/peter"] == test_dict


def test_disk_store_numpy(tmp_path):
    store = cached.DiskStore(tmp_path)
    test_list = (np.linspace(0, 1, 10, dtype=np.complex128),
                 ["one", "two", "three", "four"],
                 np.bool_(True),
                 1.234,
                 np.float64(1.2),
                 np.pi,
                 )
    store["great/bamboo"] = test_list

    meta_path = tmp_path / "great/bamboo_meta.json"
    assert meta_path.exists()

    numpy_path = tmp_path / "great/bamboo_0.npy"
    assert np.all(np.load(numpy_path) == test_list[0])
    assert np.all(store["great/bamboo"][0] == test_list[0])

    list_path = tmp_path / "great/bamboo_1.json"
    assert json.loads(list_path.read_text()) == test_list[1]
    assert np.all(store["great/bamboo"][1] == test_list[1])


def test_disk_store_path(tmp_path):
    store = cached.DiskStore(tmp_path)

    store["a/path"] = pathlib.Path(__file__)

    assert store["a/path"] == str(pathlib.Path(__file__))


def test_disk_store_remove_old_files(tmp_path):
    store = cached.DiskStore(tmp_path)

    data = np.linspace(0, 1, 1000, dtype=np.uint8)
    store["some/a"] = data
    store["some/b"] = data
    store["some/c"] = data
    store["some/d"] = data

    # This will remove the files added first
    store.remove_old_files(max_bytes=2500)

    assert "some/a" not in store
    assert "some/b" not in store
    assert "some/c" in store
    assert "some/d" in store


def test_disk_store_remove_old_files_touched(tmp_path):
    store = cached.DiskStore(tmp_path)

    data = np.linspace(0, 1, 1000, dtype=np.uint8)
    store["some/a"] = data
    store["some/b"] = data
    store["some/c"] = data
    store["some/d"] = data

    # read the first entry (refreshes cache)
    assert store["some/a"] is not None

    # This will remove the files used least
    store.remove_old_files(max_bytes=2500)

    assert "some/a" in store
    assert "some/b" not in store
    assert "some/c" not in store
    assert "some/d" in store


def test_disk_store_remove_old_files_touched_index_cleared(tmp_path):
    store = cached.DiskStore(tmp_path)

    data = np.linspace(0, 1, 1000, dtype=np.uint8)
    store["some/a"] = data
    store["some/b"] = data
    store["some/c"] = data
    store["some/d"] = data

    store.index.clear()

    # read the first entry (refreshes cache)
    assert store["some/a"] is not None

    # This will remove the files used least
    store.remove_old_files(max_bytes=2500)

    assert "some/a" in store
    assert "some/b" not in store
    assert "some/c" not in store
    assert "some/d" in store


def test_disk_store_remove_stale_locks(tmp_path):
    store = cached.DiskStore(tmp_path)

    data = np.linspace(0, 1, 1000, dtype=np.uint8)
    store["some/a"] = data
    store["some/b"] = data
    store["some/c"] = data
    store["some/d"] = data

    stale_lock = tmp_path / "some/e.lock"
    stale_lock.touch()

    # should not remove the file
    store.remove_stale_locks()
    assert stale_lock.exists()

    # set the modification time to the past. This time, lock should be removed.
    os.utime(stale_lock, (time.time(), time.time()-3601))
    store.remove_stale_locks()
    assert not stale_lock.exists()
