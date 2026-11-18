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
                    elif feat in ["volume"]:
                        # Contour data is required for computing volume.
                        # But contour data is not  exported. Instead,
                        # contour data must be computed from mask data.
                        assert np.allclose(
                            ds[feat][ecount:ecount + len(di)],
                            di[feat],
                            rtol=0,
                            atol=1e-5,
                        )
                    else:
                        assert np.all(
                            ds[feat][ecount:ecount + len(di)]
                            == di[feat]), feat
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
                for flkey in ds["trace"]:
                    trace1 = ds["trace"][flkey][ecount:ecount + len(di)]
                    trace2 = di["trace"][flkey][:]
                    assert len(trace1) == len(trace2)
                    assert np.all(trace1 == trace2), flkey
                ecount += len(di)


@pytest.mark.filterwarnings(
    "error::dclab.rtdc_dataset.feat_basin.CyclicBasinDependencyFoundWarning")
def test_split_internal_basin(tmp_path):
    """Internal basin features are stored as internal basins (#258)"""
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    path_in = tmp_path / "internal.rtdc"
    with new_dataset(h5path) as ds:
        size = len(ds)
    # four background images mapped to all events
    internal_data = {"image_bg": np.concatenate(
        [np.full((1, 80, 320), val) for val in [145, 140, 150, 142]])}
    basin_map = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 3], dtype=np.int64)
    assert size == len(basin_map)
    with h5py.File(h5path) as src, RTDCWriter(path_in) as hw:
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")
        hw.store_basin(basin_name="background",
                       basin_type="internal",
                       basin_format="h5dataset",
                       basin_locs=["basin_events"],
                       basin_descr="internal background images",
                       internal_data=internal_data,
                       basin_map=basin_map,
                       basin_feats=["image_bg"],
                       )

    paths = cli.split(path_in=path_in, path_out=tmp_path, split_events=4,
                      ret_out_paths=True)
    assert len(paths) == 3

    with new_dataset(path_in) as ds:
        image_bg = ds["image_bg"][:]

    start = 0
    # only the background images required in each output file are stored
    for pp, num_bg in zip(paths, [1, 1, 2]):
        with h5py.File(pp) as h5:
            # not stored as a regular feature
            assert "image_bg" not in h5["events"]
            assert h5["basin_events/image_bg"].shape == (num_bg, 80, 320)
        with new_dataset(pp) as ds:
            assert "image_bg" in ds.features_basin
            assert np.all(ds["image_bg"][:] == image_bg[start:start+len(ds)])
            start += len(ds)
    assert start == len(image_bg)
