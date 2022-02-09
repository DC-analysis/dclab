import h5py
import numpy as np
import pytest

import dclab
from dclab.rtdc_dataset.feat_temp import deregister_all
from dclab.rtdc_dataset.fmt_hierarchy import ChildNDArray

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


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_basic():
    """Basic test of a temporary feature"""
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
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


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_export_and_load():
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    # register temporary feature
    dclab.register_temporary_feature(feature="fl1_mean")

    with dclab.new_dataset(h5path) as ds:
        # extract the feature information from the dataset
        fl1_mean = np.array(
            [np.mean(ds["trace"]["fl1_raw"][ii]) for ii in range(len(ds))])
        # set the data
        dclab.set_temporary_feature(rtdc_ds=ds, feature="fl1_mean",
                                    data=fl1_mean)
        # export the data to a new file
        expath = h5path.with_name("exported.rtdc")
        ds.export.hdf5(expath, features=ds.features_innate + ["fl1_mean"])

    # make sure that worked
    with h5py.File(expath, "r") as h5:
        assert "fl1_mean" in h5["events"]
        assert np.allclose(h5["events"]["fl1_mean"], fl1_mean)

    # now check again with dclab
    with dclab.new_dataset(expath) as ds2:
        assert "fl1_mean" in ds2
        assert np.allclose(ds2["fl1_mean"], fl1_mean)

        # and a control check
        deregister_all()
        assert "fl1_mean" not in ds2


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_filtering():
    """Filtering with features, same example as in docs"""
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with dclab.new_dataset(h5path) as ds:
        dclab.register_temporary_feature(feature="fl1_mean")
        fl1_mean = np.array(
            [np.mean(ds["trace"]["fl1_raw"][ii]) for ii in range(len(ds))])
        dclab.set_temporary_feature(rtdc_ds=ds,
                                    feature="fl1_mean",
                                    data=fl1_mean)
        ds.config["filtering"]["fl1_mean min"] = 4
        ds.config["filtering"]["fl1_mean max"] = 200
        ds.apply_filter()
        assert np.sum(ds.filter.all) == 1
        assert ds.filter.all[1]


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_hierarchy_not_supported():
    """Test for RTDCHierarchy (does not work)"""
    # Hi there,
    # if you are here. this means that this test failed and
    # you just implemented temporary features for hierarchy
    # datasets. I'm fine with that, just make sure that the
    # root parent gets nan values.
    # Cheers,
    # Paul
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with dclab.new_dataset(h5path) as ds:
        child = dclab.new_dataset(ds)
        dclab.register_temporary_feature("my_special_feature")
        with pytest.raises(NotImplementedError):
            dclab.set_temporary_feature(rtdc_ds=child,
                                        feature="my_special_feature",
                                        data=np.arange(len(child)))


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_inherited_non_scalar():
    """Accessing inherited non-innate, non-scalar features"""
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with dclab.new_dataset(h5path) as ds:
        dclab.register_temporary_feature(feature="image_copy", is_scalar=False)
        dclab.set_temporary_feature(rtdc_ds=ds, feature="image_copy",
                                    data=ds["image"][:])
        ds.filter.manual[2] = False
        ch = dclab.new_dataset(ds)
        assert isinstance(ch["image_copy"], ChildNDArray)
        assert ch["image_copy"][:].ndim == 3
        assert isinstance(ch["image_copy"][:], np.ndarray)
        assert np.all(ch["image_copy"][2] == ds["image_copy"][3])


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_inherited_scalar():
    """Accessing inherited scalar feature should return np.ndarray"""
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with dclab.new_dataset(h5path) as ds:
        dclab.register_temporary_feature("my_special_feature")
        dclab.set_temporary_feature(rtdc_ds=ds,
                                    feature="my_special_feature",
                                    data=np.arange(len(ds)))
        ds.filter.manual[2] = False
        ch = dclab.new_dataset(ds)
        assert "my_special_feature" in ch
        assert isinstance(ch["my_special_feature"], np.ndarray)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_load_temporary_feature_from_disk():
    """Load a temporary feature from a file on disk"""
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with dclab.new_dataset(h5path) as ds:
        fl1_mean = np.array(
            [np.mean(ds["trace"]["fl1_raw"][ii]) for ii in range(len(ds))])
    with h5py.File(h5path, "a") as h5:
        h5["events"]["fl1_mean"] = fl1_mean
    # make sure it does not work without registration
    with dclab.new_dataset(h5path) as ds:
        assert "fl1_mean" not in ds, "not registered yet"
    # now, register
    dclab.register_temporary_feature(feature="fl1_mean")
    with dclab.new_dataset(h5path) as ds:
        assert "fl1_mean" in ds
        assert np.all(fl1_mean == ds["fl1_mean"])
        assert "fl1_mean" in ds._events, "registered feature loaded as usual"
        assert "fl1_mean" not in ds._usertemp, "because it is in the file"
        # make sure the feature is not available anymore when deregistered
        deregister_all()
        assert "fl1_mean" not in ds, "deregistered features are hidden..."
        assert "fl1_mean" in ds._events._features, "..but are still there"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_register_after_loading():
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with dclab.new_dataset(h5path) as ds:
        fl1_mean = np.array(
            [np.mean(ds["trace"]["fl1_raw"][ii]) for ii in range(len(ds))])
    with h5py.File(h5path, "a") as h5:
        h5["events"]["fl1_mean"] = fl1_mean
    with dclab.new_dataset(h5path) as ds:
        assert "fl1_mean" not in ds
        dclab.register_temporary_feature(feature="fl1_mean")
        assert "fl1_mean" in ds


def test_try_existing_feature_fails():
    """Basic test of a temporary feature"""
    with pytest.raises(ValueError):
        dclab.register_temporary_feature("deform")


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_with_user_config_section():
    """Use a temporary feature with the user defined config section"""
    # add some metadata to the user config section
    metadata = {"channel": True,
                "n_constrictions": 3}
    ds = dclab.new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
    ds.config["user"].update(metadata)
    assert ds.config["user"] == metadata
    area_of_region = ds["area_um"] * ds.config["user"]["n_constrictions"]

    dclab.register_temporary_feature("area_of_region")
    dclab.set_temporary_feature(rtdc_ds=ds,
                                feature="area_of_region",
                                data=area_of_region)
    area_of_region1 = ds["area_of_region"]
    area_of_region1_calc = (ds["area_um"] *
                            ds.config["user"]["n_constrictions"])
    assert np.allclose(area_of_region1, area_of_region1_calc)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_wrong_data_shape_1():
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with dclab.new_dataset(h5path) as ds:
        dclab.register_temporary_feature("my_special_feature", is_scalar=False)
        with pytest.raises(ValueError):
            dclab.set_temporary_feature(rtdc_ds=ds,
                                        feature="my_special_feature",
                                        data=np.arange(len(ds)))


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_wrong_data_shape_2():
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with dclab.new_dataset(h5path) as ds:
        dclab.register_temporary_feature("my_special_feature", is_scalar=True)
        data = np.arange(len(ds)*2).reshape(-1, 2)
        with pytest.raises(ValueError):
            dclab.set_temporary_feature(rtdc_ds=ds,
                                        feature="my_special_feature",
                                        data=data)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_wrong_length():
    """temporary feature should have same length"""
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with dclab.new_dataset(h5path) as ds:
        dclab.register_temporary_feature("my_special_feature")
        with pytest.raises(ValueError):
            dclab.set_temporary_feature(rtdc_ds=ds,
                                        feature="my_special_feature",
                                        data=np.arange(len(ds)//2))


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_wrong_name():
    """temporary feature should have same length"""
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
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
            deregister_all()
