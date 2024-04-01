import dclab
from dclab import cli, new_dataset, rtdc_dataset, RTDCWriter

import h5py
import numpy as np
import pytest

from helper_methods import retrieve_data


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_split():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    paths = cli.split(path_in=path_in, split_events=3, ret_out_paths=True)
    with new_dataset(path_in) as ds:
        ecount = 0
        for pp in paths:
            with new_dataset(pp) as di:
                for feat in ds.features_scalar:
                    if feat in ["index",
                                "time",  # issue 204
                                ]:
                        continue
                    assert np.all(
                        ds[feat][ecount:ecount + len(di)] == di[feat]), feat
                ecount += len(di)


def test_split_mapped_basin():
    """Splitting should produce mapped basins"""
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

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

    # same directory (will be cleaned up with path_in)
    path_out = h5path.parent

    # split it
    paths = cli.split(path_in=h5path_small,
                      path_out=path_out,
                      split_events=3,
                      ret_out_paths=True)

    with dclab.new_dataset(h5path) as ds0:
        # sanity check
        assert not np.all(ds0["image"][:3] == ds0["image"][3:6])
        for ii, pp in enumerate(paths):
            with dclab.new_dataset(pp) as dsi:
                # sanity check for existence of image
                assert "image" not in dsi.features_innate
                assert "image" in dsi.features_basin
                # Make sure the mapping in the split files is correct
                assert np.all(dsi["image"][:] == ds0["image"][ii*3:(ii+1)*3])


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_split_compressed():
    """Make sure the split output data are compressed"""
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.parent

    paths = cli.split(path_in=path_in,
                      path_out=path_out,
                      split_events=3,
                      ret_out_paths=True)

    for pp in paths:
        ic = rtdc_dataset.check.IntegrityChecker(pp)
        ccue = ic.check_compression()[0]
        assert ccue.data["uncompressed"] == 0


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_split_traces():
    path_in = retrieve_data("fmt-hdf5_fl_2018.zip")
    paths = cli.split(path_in=path_in, split_events=3, ret_out_paths=True)
    with new_dataset(path_in) as ds:
        ecount = 0
        for pp in paths:
            with new_dataset(pp) as di:
                for flkey in ds["trace"].keys():
                    trace1 = ds["trace"][flkey][ecount:ecount + len(di)]
                    trace2 = di["trace"][flkey][:]
                    assert len(trace1) == len(trace2)
                    assert np.all(trace1 == trace2), flkey
                ecount += len(di)
