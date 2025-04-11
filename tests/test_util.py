import importlib
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


def test_lazyloader():
    # Create a mock module
    class MockModule:
        def __init__(self):
            self.attribute = "original value"

    # Create a LazyLoader instance for the mock module
    lazy_loader = util.LazyLoader("mock_module")

    # Initially, the module should not be loaded
    assert lazy_loader._mod is None

    # Mock the importlib.import_module function
    original_import_module = importlib.import_module

    def mock_import_module(name):
        if name == "mock_module":
            return MockModule()
        return original_import_module(name)

    importlib.import_module = mock_import_module

    # Access an attribute of the lazy loader, which should trigger module
    # loading
    assert lazy_loader.attribute == "original value"

    # The module should now be loaded
    assert isinstance(lazy_loader._mod, MockModule)

    # Restore the original importlib.import_module function
    importlib.import_module = original_import_module
