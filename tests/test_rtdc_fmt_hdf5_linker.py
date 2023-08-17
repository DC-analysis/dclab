import h5py
import numpy as np

import pytest

from helper_methods import retrieve_data

from dclab import new_dataset, rtdc_dataset
from dclab.rtdc_dataset.linker import (
    ExternalDataForbiddenError,
    check_external,
    combine_h5files)


def test_linker_features_base():
    """Create a dataset that refers to a basin in a relative path"""
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Create complementary dataset
    with h5py.File(h5path, "a") as src, h5py.File(h5path_small, "w") as dst:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=dst,
                               features="scalar")
        del src["/events/deform"]
        # sanity checks
        assert "deform" in dst["events"]
        assert "deform" not in src["events"]

        assert "image" not in dst["events"]
        assert "image" in src["events"]

    fd = combine_h5files([h5path_small, h5path])

    # Now open the scalar dataset and check whether basins are defined
    with new_dataset(fd) as ds:
        assert "image" in ds
        assert "deform" in ds
        assert np.median(ds["image"][0]) == 151


def test_linker_features_error_external_links_dataset():
    """External links are forbidden"""
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_image = h5path.with_name("image.hdf5")

    # Dataset creation
    with h5py.File(h5path) as src, \
            h5py.File(h5path_image, "w") as h5:
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
        combine_h5files([h5path], external="raise")


def test_linker_features_error_external_links_group():
    """External links are forbidden"""
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_logs = h5path.with_name("logs.hdf5")

    # Dataset creation
    with h5py.File(h5path) as src, \
            h5py.File(h5path_logs, "w") as h5:
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
        combine_h5files([h5path], external="raise")


def test_linker_features_error_original_links_dataset():
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
        combine_h5files([h5path], external="raise")


def test_linker_logs():
    h5path = retrieve_data("fmt-hdf5_raw-cytoshot-exported.zip")
    fd = combine_h5files([h5path], external="raise")
    with new_dataset(fd) as ds:
        assert ds.logs
        assert len(ds.logs) == 2
        assert "so2exp_cytoshot-acquisition" in ds.logs


def test_linker_tables():
    h5path = retrieve_data("fmt-hdf5_raw-cytoshot-exported.zip")
    fd = combine_h5files([h5path], external="raise")
    with new_dataset(fd) as ds:
        assert ds.tables
        assert len(ds.tables) == 2
        assert "so2exp_cytoshot_monitor" in ds.tables
        assert np.allclose(
            ds.tables["so2exp_cytoshot_monitor"]["rotation"][0],
            2.0625)
