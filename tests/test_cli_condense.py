"""Test command-line interface dclab-condense"""
import dclab
from dclab import cli, new_dataset, util

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

    cli.condense(path_out=path_out, path_in=path_in)
    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "dclab-condense" in dsj.logs
        assert len(dsj)
        assert len(dsj) == len(ds0)
        for feat in dsj.features:
            assert np.all(dsj[feat] == ds0[feat])


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

    cli.condense(path_out=path_out, path_in=path_in)

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
                             ancillaries=True)

    with new_dataset(path_cond) as dsc:
        assert "volume" in dsc
        assert np.allclose(dsc["deform"][1000], 0.0148279015,
                           atol=0, rtol=1e-7)


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
def test_condense_triple_logs():
    """In version 0.57.6, we introduced log renaming"""
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("condensed.rtdc")
    path_out_2 = path_in.with_name("condensed2.rtdc")
    path_out_3 = path_in.with_name("condensed3.rtdc")

    cli.condense(path_out=path_out, path_in=path_in)
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

    cli.condense(path_out=path_out, path_in=path_in)
    with new_dataset(path_out) as ds:
        assert len(ds.logs) == 2
        assert "dclab-condense" in ds.logs
        assert "dclab-condense-warnings" in ds.logs
