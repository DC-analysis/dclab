import numpy as np
import pytest

import dclab
from dclab.rtdc_dataset.feat_temp import deregister_all

from helper_methods import retrieve_data


@pytest.fixture(autouse=True)
def run_around_tests():
    # Code that will run before your test, for example:
    pass
    # A test function will be run at this point
    yield
    # Code that will run after your test, for example:
    # Remove all temporary features
    deregister_all()


def test_bad_feature_name():
    """Basic test of a temporary feature"""
    with pytest.raises(ValueError):
        dclab.register_temporary_feature("Peter-Pan")


def test_basic():
    """Basic test of a temporary feature"""
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    with dclab.new_dataset(h5path) as ds:
        dclab.register_temporary_feature("my_special_feature")
        dclab.set_temporary_feature(rtdc_ds=ds,
                                    feature="my_special_feature",
                                    data=np.arange(len(ds)))
        assert ds["my_special_feature"][0] == 0


def test_basic_feature_exists():
    """Basic test of a temporary feature"""
    dclab.register_temporary_feature("my_special_feature")
    assert dclab.dfn.feature_exists("my_special_feature")


def test_hierarchy_not_supported():
    """Test for RTDCHierarchy (does not work)"""
    # Hi there,
    # if you are here. this means that this test failed and
    # you just implemented temporary features for hierarchy
    # datasets. I'm fine with that, just make sure that the
    # root parent gets nan values.
    # Cheers,
    # Paul
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    with dclab.new_dataset(h5path) as ds:
        child = dclab.new_dataset(ds)
        dclab.register_temporary_feature("my_special_feature")
        with pytest.raises(NotImplementedError):
            dclab.set_temporary_feature(rtdc_ds=child,
                                        feature="my_special_feature",
                                        data=np.arange(len(child)))


def test_try_existing_feature_fails():
    """Basic test of a temporary feature"""
    with pytest.raises(ValueError):
        dclab.register_temporary_feature("deform")


def test_wrong_length():
    """temporary feature should have same length"""
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    with dclab.new_dataset(h5path) as ds:
        dclab.register_temporary_feature("my_special_feature")
        with pytest.raises(ValueError):
            dclab.set_temporary_feature(rtdc_ds=ds,
                                        feature="my_special_feature",
                                        data=np.arange(len(ds)//2))


def test_wrong_name():
    """temporary feature should have same length"""
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    with dclab.new_dataset(h5path) as ds:
        dclab.register_temporary_feature("my_special_feature")
        with pytest.raises(ValueError):
            dclab.set_temporary_feature(rtdc_ds=ds,
                                        feature="my_other_feature",
                                        data=np.arange(len(ds)))


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
