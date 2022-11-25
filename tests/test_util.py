import pathlib
import tempfile

from dclab import new_dataset, util
import h5py
import pytest

from helper_methods import retrieve_data


@pytest.fixture(autouse=True)
def clear_cache():
    util.hashfile.cache_clear()


def test_hashfile_basic():
    td = pathlib.Path(tempfile.mkdtemp("test_hashfile"))
    p1 = td / "test_1.txt"
    p1.write_text("Lorem ipsum")
    util.hashfile(p1)
    util.hashfile(p1)
    assert util.hashfile.cache_info().misses == 1
    assert util.hashfile.cache_info().hits == 1
    assert util.hashfile.cache_info().currsize == 1

    p2 = td / "test_2.txt"
    p2.write_text("dolor sit amet.")
    util.hashfile(p2)
    assert util.hashfile.cache_info().misses == 2
    assert util.hashfile.cache_info().hits == 1
    assert util.hashfile.cache_info().currsize == 2


def test_hashfile_modified():
    td = pathlib.Path(tempfile.mkdtemp("test_hashfile"))
    p1 = td / "test_1.txt"
    p1.write_text("Lorem ipsum")
    util.hashfile(p1)
    util.hashfile(p1)
    assert util.hashfile.cache_info().misses == 1
    assert util.hashfile.cache_info().hits == 1
    assert util.hashfile.cache_info().currsize == 1

    p1.write_text("dolor sit amet.")
    util.hashfile(p1)
    assert util.hashfile.cache_info().misses == 2
    assert util.hashfile.cache_info().hits == 1
    assert util.hashfile.cache_info().currsize == 2


def test_hashfile_modified_quickly():
    td = pathlib.Path(tempfile.mkdtemp("test_hashfile"))
    p1 = td / "test_1.txt"
    p1.write_text("Lorem ipsum")
    util.hashfile(p1)
    p1.write_text("dolor sit amet.")
    util.hashfile(p1)
    assert util.hashfile.cache_info().misses == 2
    assert util.hashfile.cache_info().hits == 0
    assert util.hashfile.cache_info().currsize == 2


def test_hash_hdf5_dataset():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")

    with h5py.File(path, "r") as h5:
        hash1 = util.hashobj(h5["/events/image"])

    # sanity check
    with h5py.File(path, "r") as h5:
        hash2 = util.hashobj(h5["/events/image"])
    assert hash1 == hash2

    # now change something
    with h5py.File(path, "a") as h5:
        h5.attrs["setup:medium"] = "water"

    # actual test
    with h5py.File(path, "r") as h5:
        hash3 = util.hashobj(h5["/events/image"])
    assert hash1 != hash3


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_hash_configuration():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")

    with new_dataset(path) as ds:
        hash1 = util.hashobj(ds.config)

    # sanity check (change something unrelated)
    with h5py.File(path, "a") as h5:
        h5["events"].attrs["peter pan"] = "hook"

    with new_dataset(path) as ds:
        hash2 = util.hashobj(ds.config)
    assert hash1 == hash2

    # now the actual test
    with h5py.File(path, "a") as h5:
        h5.attrs["experiment:sample"] = "Iratrax"

    with new_dataset(path) as ds:
        hash3 = util.hashobj(ds.config)

    assert hash1 != hash3
