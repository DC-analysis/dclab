import json

import h5py
import numpy as np
import pytest

from dclab import new_dataset, rtdc_dataset, RTDCWriter
from dclab.rtdc_dataset.fmt_hdf5.basins import (
    ExternalDataForbiddenError,
    check_external,
    get_measurement_identifier,
    file_matches_identifier,
    initialize_basin_flooded_h5file)

from helper_methods import retrieve_data


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
        assert "image" in ds.features_innate
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
        assert "image" in ds.features_innate
        assert "image" in ds
        assert np.median(ds["image"][0]) == 151


def test_basin_features_error_external_links_dataset():
    """External links are forbidden"""
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_image = h5path.with_name("image.hdf5")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, \
            h5py.File(h5path_small, "w") as dst, \
            h5py.File(h5path_image, "w") as h5:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=dst,
                               features="scalar")

        # Next, store the basin information in the new dataset
        bdat = {
            "type": "file",
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
        # write image data to separate file
        h5["image"] = src["/events/image"][:]

    # turn image into an external link
    with h5py.File(h5path, "a") as src:
        del src["/events/image"]
        src["/events/image"] = h5py.ExternalLink(
            str(h5path_image), "image"
        )
        # sanity check
        assert check_external(src) == (True, "/events/image")

    with pytest.raises(ExternalDataForbiddenError,
                       match=r"not permitted for security reasons"):
        initialize_basin_flooded_h5file(h5path_small, external="raise")


def test_basin_features_error_external_links_group():
    """External links are forbidden"""
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_logs = h5path.with_name("logs.hdf5")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, \
            h5py.File(h5path_small, "w") as dst, \
            h5py.File(h5path_logs, "w") as h5:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=dst,
                               features="scalar")

        # Next, store the basin information in the new dataset
        bdat = {
            "type": "file",
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
        # write image data to separate file
        h5logs = h5.require_group("logs")
        for key in src["logs"]:
            h5logs[key] = src["logs"][key][:]

    # turn image into an external link
    with h5py.File(h5path, "a") as src:
        del src["/logs"]
        src["/logs"] = h5py.ExternalLink(
            str(h5path_logs), "/"
        )
        # sanity check
        assert check_external(src) == (True, "/logs")

    with pytest.raises(ExternalDataForbiddenError,
                       match=r"not permitted for security reasons"):
        initialize_basin_flooded_h5file(h5path_small, external="raise")


def test_basin_features_error_original_links_dataset():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_image = h5path.with_name("image.hdf5")

    # Dataset creation
    with h5py.File(h5path) as src, h5py.File(h5path_image, "w") as h5:
        # write image data to separate file
        h5["image"] = src["/events/image"][:]

    # turn image into an external link
    with h5py.File(h5path, "a") as src:
        del src["/events/image"]
        src["/events/image"] = h5py.ExternalLink(
            str(h5path_image), "image"
        )
        # sanity check
        assert check_external(src) == (True, "/events/image")

    with pytest.raises(ExternalDataForbiddenError,
                       match=r"not permitted for security reasons"):
        initialize_basin_flooded_h5file(h5path, external="raise")


def test_basin_identifiers_from_run_identifier():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    with h5py.File(h5path, "w") as hw:
        hw.attrs["experiment:run identifier"] = "hey-you-ekelpack"
    with h5py.File(h5path) as h5:
        assert get_measurement_identifier(h5) == "hey-you-ekelpack"


def test_basin_identifiers_from_time_and_setup_id():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    with h5py.File(h5path) as h5:
        assert get_measurement_identifier(h5) == \
               "01cd05a0-e084-0451-c470-2a2bf13c9e2d"


def test_basin_identifier_matches_file():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    assert file_matches_identifier(
        muid="01cd05a0-e084-0451-c470-2a2bf13c9e2d",
        path=h5path)


def test_basin_identifier_matches_file_control():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    assert not file_matches_identifier(
        muid="21cd05a0-e084-0451-c470-2a2bf13c9e2a",
        path=h5path)


def test_basin_identifier_matches_file_none():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    assert file_matches_identifier(
        muid=None,
        path=h5path)


def test_basin_identifier_matches_file_none_invalid():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    with h5py.File(h5path, "a") as h5:
        del h5.attrs["setup:identifier"]
    assert not file_matches_identifier(
        muid="01cd05a0-e084-0451-c470-2a2bf13c9e2d",
        path=h5path)
