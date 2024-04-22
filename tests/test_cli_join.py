"""Test CLI dclab-join"""
import shutil

import dclab
from dclab import cli, new_dataset, rtdc_dataset, RTDCWriter

import h5py
import numpy as np
import pytest

from helper_methods import retrieve_data


def test_join_tdms():
    pytest.importorskip("nptdms")
    path_in = retrieve_data("fmt-tdms_shapein-2.0.1-no-image_2017.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    cli.join(paths_in=[path_in, path_in], path_out=path_out)

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

    cli.join(paths_in=[path_in, path_in], path_out=path_out)

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "dclab-join" in dsj.logs
        assert "src-#1_cfg" in dsj.logs
        assert "software version = ShapeIn 2.0.1" in dsj.logs["src-#1_cfg"]
        assert "software version = ShapeIn 2.0.1" in dsj.logs["src-#2_cfg"]
        assert "src-#1_M1_camera.ini" in dsj.logs
        assert "src-#2_M1_camera.ini" in dsj.logs
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

    ret = cli.join(paths_in=[path_in, path_in], path_out=path_out)
    assert ret is None, "by default, this method should return 0 (exit 0)"
    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "dclab-join" in dsj.logs
        assert len(dsj)
        assert len(dsj) == 2 * len(ds0)
        assert np.all(dsj["circ"][:len(ds0)] == ds0["circ"])
        assert np.all(dsj["circ"][len(ds0):] == ds0["circ"])
        assert set(dsj.features) == set(ds0.features)
        assert 'identifier = ZMDD-AcC-8ecba5-cd57e2' in dsj.logs["src-#1_cfg"]
        assert 'identifier = ZMDD-AcC-8ecba5-cd57e2' in dsj.logs["src-#2_cfg"]


def test_join_rtdc_basin_data_not_written():
    """When joining data, basin features should not be written"""
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")
    h5path_out = h5path.with_name("compressed.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features=["deform", "area_um"])
        hw.store_basin(basin_name="example basin",
                       basin_type="file",
                       basin_format="hdf5",
                       basin_locs=[h5path],
                       basin_descr="an example test basin",
                       )

    # sanity check
    with new_dataset(h5path_small) as ds:
        assert "image" in ds.features

    # compress the basin-based dataset
    cli.join(paths_in=[h5path_small, h5path_small], path_out=h5path_out)

    with h5py.File(h5path_out) as h5:
        assert "basins" not in h5

    with new_dataset(h5path_out) as ds:
        assert "image" not in ds
        assert "aspect" not in ds
        assert "deform" in ds


def test_join_rtdc_basin_information_lost():
    """When joining .rtdc files, the basin information is lost

    Or at least this is not implemented in dclab 0.58.0.
    """
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")
    h5path_out = h5path.with_name("compressed.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")
        hw.store_basin(basin_name="example basin",
                       basin_type="file",
                       basin_format="hdf5",
                       basin_locs=[h5path],
                       basin_descr="an example test basin",
                       )

    # sanity check
    with new_dataset(h5path_small) as ds:
        assert "image" in ds.features

    # compress the basin-based dataset
    cli.join(paths_in=[h5path_small, h5path_small], path_out=h5path_out)

    with h5py.File(h5path_out) as h5:
        assert "basins" not in h5


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_join_rtdc_logs():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # create second file with new log
    path_in_2 = path_in.with_name("second.rtdc")
    shutil.copy2(path_in, path_in_2)
    with dclab.RTDCWriter(path_in_2, mode="append") as hw:
        hw.store_log("Dummy", [
            "One, two, three, four",
            "Everybody get on the dance floor",
            "Five, six, seven, eight",
            "It's 4 a.m. and I'm wide awake"
        ])
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    cli.join(paths_in=[path_in, path_in_2], path_out=path_out)

    with h5py.File(path_out) as h5:
        assert "src-#2_Dummy" in h5["logs"]


def test_join_rtdc_index_online_issue_158():
    """
    dclab did not correctly access events/index_online before
    """
    path1 = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    path2 = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    path_out_a = path1.with_name("outa.rtdc")

    # this did not work
    cli.join(paths_in=[path1, path2], path_out=path_out_a)

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
    cli.join(paths_in=[path1, path2], path_out=path_out_a)
    with dclab.new_dataset(path_out_a) as ds:
        # Volume is in this file, because it can be computed for path2
        assert "volume" in ds.features_innate

    # Second test: Now do the same thing with reversed dates
    with h5py.File(path2, "a") as h51:
        h51.attrs["experiment:date"] = "2019-01-01"
    cli.join(paths_in=[path1, path2], path_out=path_out_b)
    with dclab.new_dataset(path_out_b) as ds:
        assert "volume" not in ds.features_innate
        assert "dclab-join-feature-warnings" in ds.logs
        log = "\n".join(ds.logs["dclab-join-feature-warnings"])
        assert "volume" in log

    # Third test: we flip around paths_in to also test sorting
    with h5py.File(path2, "a") as h51:
        h51.attrs["experiment:date"] = "2019-01-01"
    cli.join(paths_in=[path2, path1], path_out=path_out_c)
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
    cli.join(paths_in=[path1, path2], path_out=path_out_a)
    with dclab.new_dataset(path_out_a) as ds:
        # Score cannot be computed for path2
        assert "ml_score_abc" not in ds.features_innate
        assert "dclab-join-feature-warnings" in ds.logs
        log = "\n".join(ds.logs["dclab-join-feature-warnings"])
        assert "ml_score_abc" in log

    # Second test: Now do the same thing with reversed dates
    with h5py.File(path2, "a") as h51:
        h51.attrs["experiment:date"] = "2019-01-01"
    cli.join(paths_in=[path1, path2], path_out=path_out_b)
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

    cli.join(paths_in=[path_in1, path_in2], path_out=path_out)
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
        # Necessary due to issue #204
        time = np.array(h1["events/time"], dtype=float)
        del h1["events/time"]
        h1["events/time"] = time

    with h5py.File(path_in2, mode="a") as h2:
        h2.attrs["experiment:date"] = "2019-11-05"
        h2.attrs["experiment:time"] = "16:01:15.050"
        # Necessary due to issue #204
        time = np.array(h2["events/time"], dtype=float)
        del h2["events/time"]
        h2["events/time"] = time

    offset = 24 * 60 * 60 + 60 * 60 + 1 * 60 + 15 + .05

    cli.join(paths_in=[path_in1, path_in2], path_out=path_out)
    with new_dataset(path_out) as dsj, new_dataset(path_in1) as ds0:
        assert np.allclose(dsj["time"],
                           np.concatenate((ds0["time"], ds0["time"] + offset)),
                           rtol=0,
                           atol=.0001)
