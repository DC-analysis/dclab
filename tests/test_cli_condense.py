"""Test command-line interface dclab-condense"""
from dclab import cli, new_dataset

import h5py
import numpy as np
import pytest

from helper_methods import retrieve_data


def test_check_suffix_disabled_condense():
    path_in_o = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    path_in = path_in_o.with_suffix("")
    path_in_o.rename(path_in)
    assert path_in.suffix == ""
    with pytest.raises(ValueError, match="Unsupported file type"):
        cli.condense(path_in=path_in,
                     path_out=path_in.with_name("condensed.rtdc"))
    # but this should work:
    cli.condense(path_in=path_in,
                 path_out=path_in.with_name("condensed2.rtdc"),
                 check_suffix=False)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_condense():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("condensed.rtdc")

    cli.condense(path_out=path_out, path_in=path_in)
    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "dclab-condense" in dsj.logs
        assert len(dsj)
        assert len(dsj) == len(ds0)
        for feat in dsj.features:
            assert np.all(dsj[feat] == ds0[feat])


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_condense_no_ancillary_features():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("condensed.rtdc")

    with h5py.File(path_in, "a") as h5:
        del h5["events/area_um"]

    cli.condense(path_out=path_out, path_in=path_in, ancillaries=False)
    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "volume" in ds0.features
        assert "volume" not in ds0.features_innate
        assert "contour" not in dsj.features
        assert "image" not in dsj.features
        assert "volume" not in dsj.features, "expensive feature not computed"
        assert "area_um" not in ds0.features_innate, "sanity check"
        assert "area_um" in ds0.features
        assert "area_um" in dsj.features_innate, "cheap feature available"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_condense_no_ancillary_features_control():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("condensed.rtdc")

    with h5py.File(path_in, "a") as h5:
        del h5["events/area_um"]

    cli.condense(path_out=path_out, path_in=path_in)  # defaults to True
    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "volume" in ds0.features
        assert "volume" not in ds0.features_innate
        assert "volume" in dsj.features
        assert "area_um" not in ds0.features_innate, "sanity check"
        assert "area_um" in ds0.features
        assert "area_um" in dsj.features_innate


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_condense_wo_logs():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    with h5py.File(path_in, "a") as h5:
        del h5["logs"]
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("condensed.rtdc")

    cli.condense(path_out=path_out, path_in=path_in)
    with new_dataset(path_out) as ds:
        assert len(ds.logs) == 2
        assert "dclab-condense" in ds.logs
        assert "dclab-condense-warnings" in ds.logs
