"""Test command-line interface dclab-condense"""
import dclab
from dclab import cli, new_dataset, rtdc_dataset, util, RTDCWriter

import h5py
import numpy as np
import pytest

from helper_methods import DCOR_AVAILABLE, retrieve_data


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

    ret = cli.condense(path_in=path_in, path_out=path_out)
    assert ret is None, "by default, this method should return 0 (exit 0)"
    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "dclab-condense" in dsj.logs
        assert len(dsj)
        assert len(dsj) == len(ds0)
        for feat in dsj.features:
            assert np.all(dsj[feat] == ds0[feat])


def test_condense_basins_include_by_default_data_from_basins():
    """
    When condensing a dataset, data from the basin should be
    written to the output file by default.
    """
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")
    h5path_out = h5path.with_name("condensed.rtdc")

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
        assert "aspect" in src["events"], "sanity check"

    # sanity check
    with new_dataset(h5path_small) as ds:
        assert "deform" in ds.features_innate
        assert "aspect" not in ds.features_innate
        assert "aspect" in ds.features_basin

    # condense the basin-based dataset
    cli.condense(path_in=h5path_small, path_out=h5path_out)

    with h5py.File(h5path_out) as h5:
        assert "deform" in h5["events"], "sanity check"
        assert "aspect" in h5["events"], "basin feature was copied"

    # The basin information should still be available
    with new_dataset(h5path_out) as ds:
        assert "image" in ds
        assert "image" in ds.features_basin
        assert "image" not in ds.features_innate
        assert "deform" in ds.features_innate, "sanity check"


def test_condense_basins_include_no_data_from_basins():
    """
    When condensing a dataset, data from the basin should not be
    written to the output file when `store_basin_features` is
    set to False.
    """
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")
    h5path_out = h5path.with_name("condensed.rtdc")

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
        assert "aspect" in src["events"], "sanity check"

    # sanity check
    with new_dataset(h5path_small) as ds:
        assert "deform" in ds.features_innate
        assert "aspect" not in ds.features_innate
        assert "aspect" in ds.features_basin

    # condense the basin-based dataset
    cli.condense(path_in=h5path_small, path_out=h5path_out,
                 store_ancillary_features=True,
                 store_basin_features=False)

    with h5py.File(h5path_out) as h5:
        assert "deform" in h5["events"], "sanity check"
        assert "aspect" not in h5["events"], "basin feature was copied"

    # The basin information should still be available
    with new_dataset(h5path_out) as ds:
        assert "aspect" in ds
        assert "aspect" in ds.features_basin
        assert "aspect" not in ds.features_innate
        assert "deform" in ds.features_innate, "sanity check"


@pytest.mark.parametrize("store_basins", [True, False])
def test_condense_basins_internal(store_basins):
    """
    Internal basins should just be copied to the new file
    """
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")
    h5path_out = h5path.with_name("condensed.rtdc")

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
                       internal_data={"userdef1": np.arange(2),
                                      "image_bg": np.zeros((2, 80, 320)),
                                      },
                       basin_map=np.zeros(src["events/deform"].shape[0]),
                       basin_feats=["image_bg", "userdef1"],
                       )

    # sanity check
    with new_dataset(h5path_small) as ds:
        assert "userdef1" in ds.features_basin
        assert "userdef1" not in ds.features_innate
        assert "image_bg" in ds.features_basin
        assert "image_bg" not in ds.features_innate

    # compress the basin-based dataset
    cli.condense(path_in=h5path_small, path_out=h5path_out,
                 store_basin_features=store_basins)

    with h5py.File(h5path_out) as h5:
        assert "deform" in h5["events"], "sanity check"
        # The userdef1 feature should, in any case, not be in "events",
        # because the condense step always copies basin information,
        # including any internal basins.
        assert "userdef1" not in h5["events"]
        assert "userdef1" in h5["basin_events"]
        assert np.all(h5["basin_events"]["userdef1"] == np.arange(2))
        # The image_bg feature is not scalar, so it should not be here
        assert "image_bg" not in h5["events"]
        assert "image_bg" not in h5["basin_events"]

    with new_dataset(h5path_out) as ds:
        assert "userdef1" in ds.features_basin
        assert "image_bg" in ds.features_basin


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_condense_defective_feature():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")

    # The input file has an invalid volume
    with h5py.File(path_in, "a") as h5:
        bad_volume = h5["events/volume"][:]

    # compute correct volume with dclab
    with dclab.new_dataset(path_in) as ds:
        volume = ds["volume"][:]

    # dclab computes the correct volume, identifying the volume in the
    # file as defective.
    assert not np.allclose(volume, bad_volume, atol=0, rtol=1e-3), "sanity"

    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("condensed.rtdc")

    cli.condense(path_in=path_in, path_out=path_out)

    with h5py.File(path_out) as h5o:
        volume_out = h5o["events/volume"][:]

    # The output volume should be identical to the correct volume
    assert np.allclose(volume_out, volume, atol=0, rtol=1e-10)

    # Once again with dclab
    with dclab.new_dataset(path_out) as dso:
        assert np.allclose(dso["volume"], volume, atol=0, rtol=1e-10)


@pytest.mark.skipif(not DCOR_AVAILABLE, reason="DCOR is not available")
def test_condense_from_s3(tmp_path):
    """
    dclab 0.57.6 supports condensing any class of dataset. Here
    we just test whether we can condense a resource on DCOR
    """
    pytest.importorskip("requests")
    # TODO: Upload a smaller test dataset to DCOR to speed-up this test
    s3_url = ("https://objectstore.hpccloud.mpcdf.mpg.de/"
              "circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
              "resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0")

    path_cond = tmp_path / "condensed.rtdc"
    with new_dataset(s3_url) as ds, h5py.File(path_cond, "w") as h5_cond:
        cli.condense_dataset(ds=ds,
                             h5_cond=h5_cond,
                             store_ancillary_features=True)

    with new_dataset(path_cond) as dsc:
        assert "volume" in dsc
        assert np.allclose(dsc["deform"][1000], 0.0148279015,
                           atol=0, rtol=1e-7)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_condense_include_ancillary_features():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("condensed.rtdc")

    with h5py.File(path_in, "a") as h5:
        del h5["events/area_um"]

    cli.condense(path_in=path_in,
                 path_out=path_out,
                 store_ancillary_features=False)
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

    cli.condense(path_in=path_in, path_out=path_out)  # defaults to True
    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "volume" in ds0.features
        assert "volume" not in ds0.features_innate
        assert "volume" in dsj.features
        assert "area_um" not in ds0.features_innate, "sanity check"
        assert "area_um" in ds0.features
        assert "area_um" in dsj.features_innate


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_condense_triple_logs():
    """In version 0.57.6, we introduced log renaming"""
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("condensed.rtdc")
    path_out_2 = path_in.with_name("condensed2.rtdc")
    path_out_3 = path_in.with_name("condensed3.rtdc")

    cli.condense(path_in=path_in, path_out=path_out)
    cli.condense(path_out=path_out_2, path_in=path_out)
    cli.condense(path_out=path_out_3, path_in=path_out_2)

    with dclab.new_dataset(path_out) as ds, \
        dclab.new_dataset(path_out_2) as ds2, \
            dclab.new_dataset(path_out_3) as ds3:
        md5_meta_1 = util.hashobj(ds.config)
        assert f"dclab-condense_{md5_meta_1}" in list(ds2.logs)
        assert f"dclab-condense_{md5_meta_1}" in list(ds3.logs)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_condense_wo_logs():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    with h5py.File(path_in, "a") as h5:
        del h5["logs"]
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("condensed.rtdc")

    cli.condense(path_in=path_in, path_out=path_out)
    with new_dataset(path_out) as ds:
        assert len(ds.logs) == 2
        assert "dclab-condense" in ds.logs
        assert "dclab-condense-warnings" in ds.logs
