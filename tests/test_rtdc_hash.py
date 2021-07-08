import pytest

from dclab import new_dataset


from helper_methods import example_data_dict, retrieve_data, example_data_sets


def test_hash_dict():
    ddict = example_data_dict()
    ds = new_dataset(ddict)
    assert ds.hash == "5bd08693349fd40369860474e3dab144"


def test_hash_hierarchy():
    pytest.importorskip("nptdms")
    tdms_path = retrieve_data(example_data_sets[1])
    ds1 = new_dataset(tdms_path)
    ds2 = new_dataset(ds1)
    assert ds2.hash == "3e942ba6e1cb333d3607edaba5f2c618"


def test_hash_tdms():
    pytest.importorskip("nptdms")
    tdms_path = retrieve_data(example_data_sets[1])
    ds = new_dataset(tdms_path)
    assert ds.hash == "92601489292dc9bf9fc040f87d9169c0"


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
