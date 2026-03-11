import json
import pathlib

import numpy as np
from dclab import cached

import pytest


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
