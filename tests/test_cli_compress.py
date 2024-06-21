import hashlib
import io
import json
from unittest import mock
import sys
import time

import dclab
from dclab import cli, new_dataset, rtdc_dataset, RTDCWriter

import h5py
import numpy as np
import pytest

from helper_methods import retrieve_data


def test_check_suffix_disabled_compress():
    path_in_o = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    path_in = path_in_o.with_suffix("")
    path_in_o.rename(path_in)
    assert path_in.suffix == ""
    with pytest.raises(ValueError, match="Unsupported file type"):
        cli.compress(path_in=path_in,
                     path_out=path_in.with_name("compressed.rtdc"))
    # but this should work:
    cli.compress(path_in=path_in,
                 path_out=path_in.with_name("compressed2.rtdc"),
                 check_suffix=False)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_compress():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("compressed.rtdc")

    ret = cli.compress(path_in=path_in, path_out=path_out)
    assert ret is None, "by default, this method should return 0 (exit 0)"
    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "dclab-compress" in dsj.logs
        assert len(dsj)
        assert len(dsj) == len(ds0)
        for feat in ds0.features:
            if feat in ["contour", "image", "mask"]:
                for ii in range(len(dsj)):
                    assert np.all(dsj[feat][ii] == ds0[feat][ii]), feat
            else:
                assert np.all(dsj[feat] == ds0[feat]), feat


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_compress_wo_logs():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    with h5py.File(path_in, "a") as h5:
        del h5["logs"]
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("compressed.rtdc")

    cli.compress(path_in=path_in, path_out=path_out)
    with new_dataset(path_out) as ds:
        assert len(ds.logs) == 1


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_compress_already_compressed():
    """By default, an already compressed dataset should not be compressed"""
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out1 = path_in.with_name("compressed_1.rtdc")
    path_out2 = path_in.with_name("compressed_2.rtdc")
    path_out3 = path_in.with_name("compressed_copy_of_1.rtdc")
    # this is straight-forward
    cli.compress(path_out=path_out1, path_in=path_in)
    # just for the sake of comparison
    time.sleep(1)  # we need different time stamps in path_out2
    cli.compress(path_out=path_out2, path_in=path_in)
    # this is not trivial
    cli.compress(path_out=path_out3, path_in=path_out1)

    # the first two files should not be the same (dates are written, etc)
    h1 = hashlib.md5(path_out1.read_bytes()).hexdigest()
    h2 = hashlib.md5(path_out2.read_bytes()).hexdigest()
    h3 = hashlib.md5(path_out3.read_bytes()).hexdigest()
    assert h1 != h2
    # Changed in dclab 0.49.0: Since the compression step should also check
    # for defective features, it is important to revisit the entire file.
    # As such, it is cleaner to rewrite the entire dataset, since we can now
    # copy single HDF5 Datasets without having to redo the compression.
    # assert h1 == h3
    assert h1 != h3


def test_compress_basin_internal():
    """
    Internal basins should just be copied to the new file
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
                       basin_type="internal",
                       basin_format="h5dataset",
                       basin_locs=["basin_events"],
                       basin_descr="an example test basin",
                       internal_data={"userdef1": np.arange(2)},
                       basin_map=np.zeros(src["events/deform"].shape[0]),
                       basin_feats=["userdef1"],
                       )

    # sanity check
    with new_dataset(h5path_small) as ds:
        assert "userdef1" in ds.features_basin
        assert "userdef1" not in ds.features_innate

    # compress the basin-based dataset
    cli.compress(path_in=h5path_small, path_out=h5path_out)

    with h5py.File(h5path_out) as h5:
        assert "deform" in h5["events"], "sanity check"
        assert "userdef1" not in h5["events"]
        assert "userdef1" in h5["basin_events"]
        assert np.all(h5["basin_events"]["userdef1"] == np.arange(2))

    with new_dataset(h5path_out) as ds:
        assert "userdef1" in ds.features_basin


def test_compress_basin_no_data_from_basins():
    """
    When compressing a dataset, feature data from the basin should not be
    written to the output file.
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
    cli.compress(path_in=h5path_small, path_out=h5path_out)

    with h5py.File(h5path_out) as h5:
        assert "deform" in h5["events"], "sanity check"
        assert "image" not in h5["events"], "Arrgh, basin feature was copied"

    with new_dataset(h5path_out) as ds:
        assert "image" in ds.features_basin


def test_compress_basin_preserved_compress():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Empty, basin-based dataset creation
    with RTDCWriter(h5path_small) as hw:
        bn_hash = hw.store_basin(basin_name="example basin",
                                 basin_type="file",
                                 basin_format="hdf5",
                                 basin_locs=[h5path],
                                 basin_descr="an example test basin",
                                 )
    # compress the data
    h5path_out = h5path_small.with_name("compressed.rtdc")
    cli.compress(path_in=h5path_small,
                 path_out=h5path_out,
                 )

    with h5py.File(h5path_out) as h5_out, h5py.File(h5path_small) as h5_in:
        # check if h5path_out is empty
        if 'events' in h5_out:
            assert len(h5_out['events']) == 0
        # check if h5path_small basin is same as h5path_out basin
        assert 'basins' in h5_in
        assert 'basins' in h5_out
        assert np.all(
            h5_in['basins'][bn_hash][:] == h5_out['basins'][bn_hash][:]
        )

    with new_dataset(h5path) as ds, new_dataset(h5path_out) as ds_out:
        # check if all features of h5path are preserved in h5path_out
        assert len(ds.features_innate) == 32
        assert len(ds_out.features_innate) == 0
        for feat in ds.features_innate:
            assert feat in ds_out.features_basin


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_compress_log_md5_5m():
    """In dclab 0.42.0 we changed sha256 to md5-5M file checksums"""
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # sanity check (file is < 5MB)
    h1 = hashlib.md5(path_in.read_bytes()).hexdigest()
    assert h1 == "e49db02274ac75ab24911f893c41f5b0"
    # same directory (will be cleaned up with path_in)
    path_out1 = path_in.with_name("compressed_1.rtdc")
    cli.compress(path_out=path_out1, path_in=path_in)
    with dclab.new_dataset(path_out1) as ds:
        log = ds.logs["dclab-compress"]
    dcdict = json.loads("\n".join(log))
    file = dcdict["files"][0]
    assert file["index"] == 1
    assert file["name"] == "mask_contour_reference.rtdc"
    assert file["md5-5M"] == "e49db02274ac75ab24911f893c41f5b0"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_compress_with_online_polygon_filters():
    """Shape-In 2.3 supports online polygon filters"""
    path = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # add an artificial online polygon filter
    with h5py.File(path, "a") as h5:
        # set soft filter to True
        h5.attrs["online_filter:area_um,deform soft limit"] = True
        # set filter values
        pf_name = "online_filter:area_um,deform polygon points"
        area_um = h5["events"]["area_um"]
        deform = h5["events"]["deform"]
        pf_points = np.array([
            [np.mean(area_um) + np.std(area_um),
             np.mean(deform)],
            [np.mean(area_um) + np.std(area_um),
             np.mean(deform) + np.std(deform)],
            [np.mean(area_um),
             np.mean(deform) + np.std(deform)],
        ])
        h5.attrs[pf_name] = pf_points

    path_out = path.with_name("compressed.rtdc")
    cli.compress(path_out=path_out, path_in=path)

    with dclab.new_dataset(path_out) as ds:
        assert len(ds) == 8
        assert ds.config["online_filter"]["area_um,deform soft limit"]
        assert "area_um,deform polygon points" in ds.config["online_filter"]
        assert np.allclose(
            ds.config["online_filter"]["area_um,deform polygon points"],
            pf_points)


def test_compress_with_online_polygon_filters_real_data():
    """Shape-In 2.3 supports online polygon filters"""
    path = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")

    path_out = path.with_name("compressed.rtdc")
    cli.compress(path_out=path_out, path_in=path)

    with dclab.new_dataset(path_out) as ds:
        assert len(ds) == 1
        assert ds.config["online_filter"]["size_x,size_y soft limit"]
        assert "size_x,size_y polygon points" in ds.config["online_filter"]
        assert np.allclose(
            ds.config["online_filter"]["size_x,size_y polygon points"],
            [[0.1, 0.2],
             [0.1, 2.5],
             [3.3, 3.2],
             [5.2, 0.9]]
        )


@mock.patch('sys.stdout', new_callable=io.StringIO)
def test_version(mock_stdout, monkeypatch):
    def sys_exit(status):
        return status
    monkeypatch.setattr(sys, "exit", sys_exit)

    monkeypatch.setattr(sys, "argv", ["dclab-compress", "--version"])

    parser = cli.compress_parser()
    parser.parse_args()

    stdout_printed = mock_stdout.getvalue()
    assert stdout_printed.count("dclab-compress")
    assert stdout_printed.count(dclab.__version__)
