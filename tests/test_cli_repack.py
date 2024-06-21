from dclab import cli, new_dataset, rtdc_dataset, RTDCWriter

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

    ret = cli.repack(path_in=path_in, path_out=path_out)
    assert ret is None, "by default, this method should return 0 (exit 0)"

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


@pytest.mark.parametrize("use_basins", [True, False])
def test_repack_basin_internal(use_basins):
    """
    Internal basins should just be copied to the new file
    """
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")
    h5path_out = h5path.with_name("repacked.rtdc")

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
    cli.repack(path_in=h5path_small, path_out=h5path_out,
               strip_basins=not use_basins)

    with h5py.File(h5path_out) as h5:
        assert "deform" in h5["events"], "sanity check"
        # userdef1 should not be in "events" in any case
        assert "userdef1" not in h5["events"]
        if use_basins:
            assert "userdef1" in h5["basin_events"]
            assert np.all(h5["basin_events"]["userdef1"] == np.arange(2))
        else:
            assert "basin_events" not in h5

    with new_dataset(h5path_out) as ds:
        if use_basins:
            assert "userdef1" in ds.features_basin
        else:
            assert "userdef1" not in ds.features_basin


def test_repack_no_data_from_basins_written():
    """
    When repacking a dataset, feature data from the basin should not be
    written to the output file.
    """
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")
    h5path_out = h5path.with_name("repacked.rtdc")

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

    # repack the basin-based dataset
    cli.repack(path_in=h5path_small, path_out=h5path_out)

    with h5py.File(h5path_out) as h5:
        assert "deform" in h5["events"], "sanity check"
        assert "image" not in h5["events"], "Arrgh, basin feature was copied"

    with new_dataset(h5path_out) as ds:
        assert "image" in ds.features_basin


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
def test_repack_strip_basins():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")
    h5path_out = h5path.with_name("repacked.rtdc")

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

    # repack the basin-based dataset
    cli.repack(path_in=h5path_small,
               path_out=h5path_out,
               strip_basins=True)

    with h5py.File(h5path_out) as h5:
        assert "deform" in h5["events"], "sanity check"
        assert "image" not in h5["events"], "Arrgh, basin feature was copied"

    with new_dataset(h5path_out) as ds:
        assert "image" not in ds.features_basin, "[sic] we wanted to strip"


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
