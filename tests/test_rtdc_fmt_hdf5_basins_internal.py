"""Internal basins were introduced in version 0.60.0

They provide a few-to-many mapping for feature data and can be used
to achieve high compression ratios, an advantage that is primarily
of interest for the "image_bg" feature.
"""
import h5py
import numpy as np

from dclab import new_dataset, rtdc_dataset, RTDCWriter

import pytest

from helper_methods import retrieve_data


@pytest.mark.parametrize("mapping", [
    # monotonic
    np.array([0] * 2 + [1] * 3 + [2] * 1 + [3] * 4),
    # irregular access (useful for e.g. temperature)
    np.array([1, 0, 2, 3, 0, 2, 0, 1, 3, 1]),
])
def test_read_write_internal_basin_data(mapping):
    """write internal basin data and make sure everything is correct"""
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    internal_data = {
        "userdef1": np.arange(4),
        "image_bg": np.concatenate([
            np.full((1, 80, 320), 145),
            np.full((1, 80, 320), 140),
            np.full((1, 80, 320), 150),
            np.full((1, 80, 320), 142),
            ])
    }

    # sanity checks
    with h5py.File(h5path) as h5:
        assert len(h5["events/deform"]) == 10
        assert len(h5["events/deform"]) == len(mapping)

    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")
        # second, define the internal basin
        hw.store_basin(basin_name="example basin",
                       basin_type="internal",
                       basin_format="h5dataset",
                       basin_locs=["basin_events"],
                       basin_descr="an example test basin",
                       internal_data=internal_data,
                       basin_map=mapping,
                       basin_feats=sorted(internal_data.keys()),
                       )

    # sanity checks
    with h5py.File(h5path_small) as h5:
        assert np.all(h5["basin_events/userdef1"][:]
                      == internal_data["userdef1"])
        assert np.all(h5["basin_events/image_bg"][:]
                      == internal_data["image_bg"])

    # now try to open the dataset containing the internal basin
    with new_dataset(h5path_small) as ds:
        for ii, mid in enumerate(mapping):
            assert np.all(ds["image_bg"][ii]
                          == internal_data["image_bg"][mid])
            assert np.all(ds["userdef1"][ii]
                          == internal_data["userdef1"][mid])


@pytest.mark.parametrize("mapping", [
    # monotonic
    np.array([0] * 2 + [1] * 3 + [2] * 1 + [3] * 4),
    # irregular access (useful for e.g. temperature)
    np.array([1, 0, 2, 3, 0, 2, 0, 1, 3, 1]),
])
def test_read_write_export_internal_basin_data_issue_262(mapping):
    """write internal basin data and make sure everything is correct"""
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")
    h5path_export = h5path.with_name("exported.rtdc")

    internal_data = {
        "userdef1": np.arange(4),
        "image_bg": np.concatenate([
            np.full((1, 80, 320), 145),
            np.full((1, 80, 320), 140),
            np.full((1, 80, 320), 150),
            np.full((1, 80, 320), 142),
        ])
    }

    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")
        # second, define the internal basin
        hw.store_basin(basin_name="example basin",
                       basin_type="internal",
                       basin_format="h5dataset",
                       basin_locs=["basin_events"],
                       basin_descr="an example test basin",
                       internal_data=internal_data,
                       basin_map=mapping,
                       basin_feats=sorted(internal_data.keys()),
                       )

    # now try to open the dataset containing the internal basin
    # and export it to another .rtdc file.
    with new_dataset(h5path_small) as ds:
        ds.export.hdf5(
            path=h5path_export,
            features=["userdef1", "deform"],
            logs=True,
            tables=True,
            basins=True,  # this is tested in issue #262
            meta_prefix="",
            override=False)
        assert len(ds["deform"]) == 10
        assert len(ds["userdef1"][:]) == 10
        assert len(ds["userdef1"]) == 10

    with new_dataset(h5path_export) as dse:
        assert "image_bg" in dse.features_basin
        assert "image_bg" not in dse.features_innate
        assert np.all(dse["userdef1"] == internal_data["userdef1"][mapping])


def test_writer_error_internal_data():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")
        with pytest.raises(ValueError,
                           match="you must specify `internal_data`"):
            hw.store_basin(
                basin_name="example basin",
                basin_type="internal",
                basin_format="h5dataset",
                basin_locs=["basin_events"],
                basin_descr="an example test basin",
                basin_map=np.zeros(10),
                basin_feats=["userdef1"],
                )


def test_writer_error_basin_data():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")

        with pytest.raises(ValueError,
                           match="Could not find feature 'userdef1'"):
            hw.store_basin(
                basin_name="example basin",
                basin_type="internal",
                basin_format="h5dataset",
                basin_locs=["basin_events"],
                basin_descr="an example test basin",
                basin_map=np.zeros(10),
                basin_feats=["userdef1"],
                internal_data=hw.h5file.require_group("basin_events")
                )


def test_writer_error_feature_exists():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")
        igroup = hw.h5file.require_group("basin_events")
        hw.write_ndarray(igroup, "userdef1", np.arange(2))

        with pytest.raises(ValueError,
                           match="feature 'userdef1' already exists"):
            hw.store_basin(
                basin_name="example basin",
                basin_type="internal",
                basin_format="h5dataset",
                basin_locs=["basin_events"],
                basin_descr="an example test basin",
                basin_map=np.zeros(10),
                basin_feats=["userdef1"],
                internal_data={"userdef1": np.arange(2)},
                )


def test_writer_warning_uncommon_location():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")
        internal_data = hw.h5file.require_group("uncommon_loc")
        hw.write_ndarray(internal_data,
                         name="userdef1",
                         data=np.arange(2),
                         )
        with pytest.warns(UserWarning,
                          match="You specified an uncommon location"):
            hw.store_basin(
                basin_name="example basin",
                basin_type="internal",
                basin_format="h5dataset",
                basin_locs=["uncommon_loc"],
                basin_descr="an example test basin",
                basin_map=np.zeros(10),
                basin_feats=["userdef1"],
                internal_data=internal_data,
                )
