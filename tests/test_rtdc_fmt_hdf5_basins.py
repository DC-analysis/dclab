import json

import h5py
import numpy as np

import pytest

import dclab
from dclab import new_dataset, rtdc_dataset, RTDCWriter
from dclab.rtdc_dataset.fmt_hdf5.base import HDF5Basin


from helper_methods import retrieve_data


def test_basin_not_available():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, h5py.File(h5path_small, "w") as dst:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=dst,
                               features="scalar")
        # Next, store the basin information in the new dataset
        bdat = {
            "type": "file",
            "format": "hdf5",
            "paths": [
                "fake.rtdc",  # fake path
                str(h5path),  # absolute path name
            ]
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = dst.require_group("basins")
        with RTDCWriter(dst, mode="append") as hw:
            hw.write_text(basins, "my_basin", blines)
        # sanity checks
        assert "deform" in dst["events"]
        assert "image" not in dst["events"]

    h5path.unlink()

    # Now open the scalar dataset and check whether basins missing
    with new_dataset(h5path_small) as ds:
        assert "image" not in ds
        assert not ds.features_basin

    # Also test that on a lower level
    bn = HDF5Basin(h5path)
    assert not bn.is_available()
    with pytest.raises(ValueError, match="is not available"):
        _ = bn.ds


def test_basin_features_path_absolute():
    """Create a dataset that refers to a basin in a relative path"""
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, h5py.File(h5path_small, "w") as dst:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=dst,
                               features="scalar")
        # Next, store the basin information in the new dataset
        bdat = {
            "type": "file",
            "format": "hdf5",
            "paths": [
                "fake.rtdc",  # fake path
                str(h5path),  # absolute path name
            ]
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = dst.require_group("basins")
        with RTDCWriter(dst, mode="append") as hw:
            hw.write_text(basins, "my_basin", blines)
        # sanity checks
        assert "deform" in dst["events"]
        assert "image" not in dst["events"]

    # Now open the scalar dataset and check whether basins are defined
    with new_dataset(h5path_small) as ds:
        assert "image" in ds.features_basin
        assert "image" not in ds.features_innate
        assert "image" in ds
        assert np.median(ds["image"][0]) == 151


def test_basin_features_path_relative():
    """Create a dataset that refers to a basin in a relative path"""
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, h5py.File(h5path_small, "w") as dst:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=dst,
                               features="scalar")
        # Next, store the basin information in the new dataset
        bdat = {
            "type": "file",
            "format": "hdf5",
            "paths": [
                "fake.rtdc",  # fake path
                h5path.name,  # relative path name
            ]
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = dst.require_group("basins")
        with RTDCWriter(dst, mode="append") as hw:
            hw.write_text(basins, "my_basin", blines)
        # sanity checks
        assert "deform" in dst["events"]
        assert "image" not in dst["events"]

    # Now open the scalar dataset and check whether basins are defined
    with new_dataset(h5path_small) as ds:
        assert "image" in ds.features_basin
        assert "image" not in ds.features_innate
        assert "image" in ds
        assert np.median(ds["image"][0]) == 151


def test_basin_identifiers_from_run_identifier():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    with h5py.File(h5path, "a") as hw:
        hw.attrs["experiment:run identifier"] = "hey-you-ekelpack"
    with dclab.new_dataset(h5path) as ds:
        assert ds.get_measurement_identifier() == "hey-you-ekelpack"


def test_basin_identifiers_from_time_and_setup_id():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    with dclab.new_dataset(h5path) as ds:
        assert ds.get_measurement_identifier() == \
               "01cd05a0-e084-0451-c470-2a2bf13c9e2d"


def test_basin_identifier_matches_file_none_invalid():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    with h5py.File(h5path, "a") as h5:
        del h5.attrs["setup:identifier"]
    with dclab.new_dataset(h5path) as ds:
        assert not ds.get_measurement_identifier() == \
               "01cd05a0-e084-0451-c470-2a2bf13c9e2d"
