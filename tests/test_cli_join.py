"""Test CLI dclab-join"""
from dclab import cli, new_dataset

import h5py
import numpy as np
import pytest

from helper_methods import retrieve_data


def test_join_tdms():
    pytest.importorskip("nptdms")
    path_in = retrieve_data("fmt-tdms_shapein-2.0.1-no-image_2017.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    cli.join(path_out=path_out, paths_in=[path_in, path_in])

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert len(dsj)
        assert len(dsj) == 2 * len(ds0)
        assert len(ds0) == ds0.config["experiment"]["event count"]
        assert len(dsj) == dsj.config["experiment"]["event count"]
        assert np.all(dsj["circ"][:100] == ds0["circ"][:100])
        assert np.all(dsj["circ"][len(ds0):len(ds0)+100] == ds0["circ"][:100])
        assert set(dsj.features) == set(ds0.features)


def test_join_tdms_logs():
    pytest.importorskip("nptdms")
    path_in = retrieve_data("fmt-tdms_shapein-2.0.1-no-image_2017.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    cli.join(path_out=path_out, paths_in=[path_in, path_in])

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "dclab-join" in dsj.logs
        assert "cfg-#1" in dsj.logs
        assert "software version = ShapeIn 2.0.1" in dsj.logs["cfg-#1"]
        assert ds0.logs
        for key in ds0.logs:
            jkey = "src-#1_" + key
            assert np.all(np.array(ds0.logs[key]) == np.array(dsj.logs[jkey]))


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_join_rtdc():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    cli.join(path_out=path_out, paths_in=[path_in, path_in])
    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "dclab-join" in dsj.logs
        assert len(dsj)
        assert len(dsj) == 2 * len(ds0)
        assert np.all(dsj["circ"][:len(ds0)] == ds0["circ"])
        assert np.all(dsj["circ"][len(ds0):] == ds0["circ"])
        assert set(dsj.features) == set(ds0.features)
        assert 'identifier = ZMDD-AcC-8ecba5-cd57e2' in dsj.logs["cfg-#1"]


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_join_frame():
    path_in1 = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    path_in2 = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in1.with_name("out.rtdc")

    # modify acquisition times
    with h5py.File(path_in1, mode="a") as h1:
        h1.attrs["experiment:date"] = "2019-11-04"
        h1.attrs["experiment:time"] = "15:00:00"

    with h5py.File(path_in2, mode="a") as h2:
        h2.attrs["experiment:date"] = "2019-11-05"
        h2.attrs["experiment:time"] = "16:01:15.050"

    offset = 24 * 60 * 60 + 60 * 60 + 1 * 60 + 15 + .05

    cli.join(path_out=path_out, paths_in=[path_in1, path_in2])
    with new_dataset(path_out) as dsj, new_dataset(path_in1) as ds0:
        fr = ds0.config["imaging"]["frame rate"]
        assert np.allclose(dsj["frame"],
                           np.concatenate((ds0["frame"],
                                           ds0["frame"] + offset * fr)),
                           rtol=0,
                           atol=.0001)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_join_times():
    path_in1 = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    path_in2 = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in1.with_name("out.rtdc")

    # modify acquisition times
    with h5py.File(path_in1, mode="a") as h1:
        h1.attrs["experiment:date"] = "2019-11-04"
        h1.attrs["experiment:time"] = "15:00:00"

    with h5py.File(path_in2, mode="a") as h2:
        h2.attrs["experiment:date"] = "2019-11-05"
        h2.attrs["experiment:time"] = "16:01:15.050"

    offset = 24 * 60 * 60 + 60 * 60 + 1 * 60 + 15 + .05

    cli.join(path_out=path_out, paths_in=[path_in1, path_in2])
    with new_dataset(path_out) as dsj, new_dataset(path_in1) as ds0:
        assert np.allclose(dsj["time"],
                           np.concatenate((ds0["time"], ds0["time"] + offset)),
                           rtol=0,
                           atol=.0001)
