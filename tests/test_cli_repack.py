from dclab import cli, new_dataset, rtdc_dataset

import h5py
import numpy as np
import pytest

from helper_methods import retrieve_data


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_repack_basic():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("repacked.rtdc")

    cli.repack(path_in=path_in, path_out=path_out)

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert len(dsj)
        assert len(dsj) == len(ds0)
        for feat in ds0.features_innate:
            if feat in ds0.features_scalar:
                assert np.all(dsj[feat] == ds0[feat]), feat
        for ii in range(len(ds0)):
            assert np.all(dsj["contour"][ii] == ds0["contour"][ii])
            assert np.all(dsj["image"][ii] == ds0["image"][ii])
            assert np.all(dsj["mask"][ii] == ds0["mask"][ii])


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_repack_remove_secrets():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("repacked.rtdc")

    with h5py.File(path_in, "a") as h5:
        h5.attrs["experiment:sample"] = "my dirty secret"

    with h5py.File(path_in, "a") as h5:
        h5.attrs["experiment:sample"] = "sunshine"

    # test whether the dirty secret is still there
    with open(str(path_in), "rb") as fd:
        data = fd.read()
        assert str(data).count("my dirty secret")

    # now repack
    cli.repack(path_in=path_in, path_out=path_out)

    # clean?
    with open(str(path_out), "rb") as fd:
        data = fd.read()
        assert not str(data).count("my dirty secret")


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_repack_strip_logs():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("repacked.rtdc")

    # write some logs
    with h5py.File(path_in, "a") as h5:
        hw = rtdc_dataset.RTDCWriter(h5)
        hw.store_log("test_log", ["peter", "hans"])

    cli.repack(path_in=path_in, path_out=path_out, strip_logs=True)

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert ds0.logs
        assert not dsj.logs


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_repack_user_metadata():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    with h5py.File(path_in, "a") as h5:
        h5.attrs["user:peter"] = "hans"

    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("repacked.rtdc")

    cli.repack(path_in=path_in, path_out=path_out)

    with new_dataset(path_out) as ds:
        assert ds.config["user"]["peter"] == "hans"