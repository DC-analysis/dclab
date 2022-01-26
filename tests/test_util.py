import pathlib
import tempfile

from dclab import util
import pytest


@pytest.fixture(autouse=True)
def clear_cache():
    util._hashfile_cached.cache_clear()


def test_hashfile_basic():
    td = pathlib.Path(tempfile.mkdtemp("test_hashfile"))
    p1 = td / "test_1.txt"
    p1.write_text("Lorem ipsum")
    util.hashfile(p1)
    util.hashfile(p1)
    assert util._hashfile_cached.cache_info().misses == 1
    assert util._hashfile_cached.cache_info().hits == 1
    assert util._hashfile_cached.cache_info().currsize == 1

    p2 = td / "test_2.txt"
    p2.write_text("dolor sit amet.")
    util.hashfile(p2)
    assert util._hashfile_cached.cache_info().misses == 2
    assert util._hashfile_cached.cache_info().hits == 1
    assert util._hashfile_cached.cache_info().currsize == 2


def test_hashfile_modified():
    td = pathlib.Path(tempfile.mkdtemp("test_hashfile"))
    p1 = td / "test_1.txt"
    p1.write_text("Lorem ipsum")
    util.hashfile(p1)
    util.hashfile(p1)
    assert util._hashfile_cached.cache_info().misses == 1
    assert util._hashfile_cached.cache_info().hits == 1
    assert util._hashfile_cached.cache_info().currsize == 1

    p1.write_text("dolor sit amet.")
    util.hashfile(p1)
    assert util._hashfile_cached.cache_info().misses == 2
    assert util._hashfile_cached.cache_info().hits == 1
    assert util._hashfile_cached.cache_info().currsize == 2


def test_hashfile_modified_quickly():
    td = pathlib.Path(tempfile.mkdtemp("test_hashfile"))
    p1 = td / "test_1.txt"
    p1.write_text("Lorem ipsum")
    util.hashfile(p1)
    p1.write_text("dolor sit amet.")
    util.hashfile(p1)
    assert util._hashfile_cached.cache_info().misses == 2
    assert util._hashfile_cached.cache_info().hits == 0
    assert util._hashfile_cached.cache_info().currsize == 2
