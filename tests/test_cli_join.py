"""Test CLI dclab-join"""
import dclab
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


def test_join_rtdc_index_online_issue_158():
    """
    dclab did not correctly access events/index_online before
    """
    path1 = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    path2 = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    path_out_a = path1.with_name("outa.rtdc")

    # this did not work
    cli.join(path_out=path_out_a, paths_in=[path1, path2])

    # verification
    with dclab.new_dataset(path_out_a) as ds:
        assert "index_online" in ds.features_innate
        assert np.all(np.diff(ds["index_online"]) > 0)


def test_join_rtdc_unequal_features_issue_157():
    """
    If two files do not contain the same number of features, joining
    should only take into account the same features and then should
    issue a warning (which will then be written to the logs in DCKit).
    """
    path1 = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    path2 = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    path_out_a = path1.with_name("outa.rtdc")
    path_out_b = path1.with_name("outb.rtdc")
    path_out_c = path1.with_name("outc.rtdc")

    # add a feature to path1 and define order (join sorts the files with date)
    with h5py.File(path1, "a") as h51:
        h51["events"]["volume"] = h51["events"]["area_um"][:] ** 1.5
        h51.attrs["experiment:date"] = "2020-01-01"
    with h5py.File(path2, "a") as h51:
        h51.attrs["experiment:date"] = "2021-01-01"

    # sanity checks
    with dclab.new_dataset(path1) as ds:
        assert "volume" in ds.features_innate
    with dclab.new_dataset(path2) as ds:
        assert "volume" not in ds.features_innate

    # First test
    # There should be no warning here, because for path2 volume can be
    # computed.
    cli.join(path_out=path_out_a, paths_in=[path1, path2])
    with dclab.new_dataset(path_out_a) as ds:
        # Volume is in this file, because it can be computed for path2
        assert "volume" in ds.features_innate

    # Second test: Now do the same thing with reversed dates
    with h5py.File(path2, "a") as h51:
        h51.attrs["experiment:date"] = "2019-01-01"
    cli.join(path_out=path_out_b, paths_in=[path1, path2])
    with dclab.new_dataset(path_out_b) as ds:
        assert "volume" not in ds.features_innate
        assert "dclab-join-feature-warnings" in ds.logs
        log = "\n".join(ds.logs["dclab-join-feature-warnings"])
        assert "volume" in log

    # Third test: we flip around paths_in to also test sorting
    with h5py.File(path2, "a") as h51:
        h51.attrs["experiment:date"] = "2019-01-01"
    cli.join(path_out=path_out_c, paths_in=[path2, path1])
    with dclab.new_dataset(path_out_c) as ds:
        assert "volume" not in ds.features_innate
        assert "dclab-join-feature-warnings" in ds.logs
        log = "\n".join(ds.logs["dclab-join-feature-warnings"])
        assert "volume" in log


def test_join_rtdc_unequal_features_issue_157_2():
    """
    Same test as above, but we use a feature that cannot be computed
    """
    path1 = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    path2 = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    path_out_a = path1.with_name("outa.rtdc")
    path_out_b = path1.with_name("outb.rtdc")

    # add a feature to path1 and define order (join sorts the files with date)
    with h5py.File(path1, "a") as h51:
        h51["events"]["ml_score_abc"] = np.log10(h51["events"]["area_um"][:])
        h51.attrs["experiment:date"] = "2020-01-01"
    with h5py.File(path2, "a") as h51:
        h51.attrs["experiment:date"] = "2021-01-01"

    # sanity checks
    with dclab.new_dataset(path1) as ds:
        assert "ml_score_abc" in ds.features_innate
    with dclab.new_dataset(path2) as ds:
        assert "ml_score_abc" not in ds.features_innate

    # First test
    cli.join(path_out=path_out_a, paths_in=[path1, path2])
    with dclab.new_dataset(path_out_a) as ds:
        # Score cannot be computed for path2
        assert "ml_score_abc" not in ds.features_innate
        assert "dclab-join-feature-warnings" in ds.logs
        log = "\n".join(ds.logs["dclab-join-feature-warnings"])
        assert "ml_score_abc" in log

    # Second test: Now do the same thing with reversed dates
    with h5py.File(path2, "a") as h51:
        h51.attrs["experiment:date"] = "2019-01-01"
    cli.join(path_out=path_out_b, paths_in=[path1, path2])
    with dclab.new_dataset(path_out_b) as ds:
        assert "ml_score_abc" not in ds.features_innate
        assert "dclab-join-feature-warnings" in ds.logs
        log = "\n".join(ds.logs["dclab-join-feature-warnings"])
        assert "ml_score_abc" in log


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
