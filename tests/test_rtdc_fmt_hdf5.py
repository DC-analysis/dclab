"""Test hdf5 file format"""
import io
import os
import pathlib
import shutil
import sys
import tempfile

import h5py
import numpy as np
import pytest

import dclab
from dclab import new_dataset, rtdc_dataset
from dclab.rtdc_dataset import config, rtdc_copy

from helper_methods import retrieve_data


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_config():
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2017.zip"))
    assert ds.config["setup"]["channel width"] == 30
    assert ds.config["setup"]["chip region"].lower() == "channel"
    assert ds.config["setup"]["flow rate"] == 0.16
    assert ds.config["imaging"]["pixel size"] == 0.34


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_contour_basic():
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2017.zip"))
    assert len(ds) == 5
    assert len(ds["contour"]) == 5
    assert ds["deform"].ndim == 1  # important for matplotlib
    assert np.allclose(np.average(ds["contour"][0]), 30.75)
    assert np.median(ds["contour"][0]) == 30.5
    assert np.median(ds["contour"][2]) == 31.5
    assert np.median(ds["contour"][-3]) == 31.5
    assert np.median(ds["contour"][4]) == 33.0
    assert np.median(ds["contour"][-1]) == 33.0


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_defective_feature_aspect():
    # see https://github.com/ZELLMECHANIK-DRESDEN/ShapeOut/issues/241
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    # modify aspect feature
    with h5py.File(h5path, "a") as h5:
        aspect0 = h5["events/aspect"][0]
        aspect1 = 1.234
        assert not np.allclose(aspect0, aspect1), "test's sanity"
        h5["events/aspect"][0] = aspect1
        # In Shape-In 2.0.5 everything worked fine
        h5.attrs["setup:software version"] = "ShapeIn 2.0.5"
    # sanity check
    with new_dataset(h5path) as ds1:
        assert np.allclose(ds1["aspect"][0], aspect1)
    # trigger recomputation of aspect feature
    with h5py.File(h5path, "a") as h5:
        h5.attrs["setup:software version"] = "ShapeIn 2.0.6"
    # verify original value of aspect
    with new_dataset(h5path) as ds2:
        assert np.allclose(ds2["aspect"][0], aspect0)


@pytest.mark.parametrize("feat", [
    "inert_ratio_cvx",
    "inert_ratio_prnc",
    "inert_ratio_raw",
    "tilt"])
def test_defective_feature_inert_ratio_prnc(feat):
    # see https://github.com/DC-analysis/dclab/issues/212
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    # sanity check
    with h5py.File(h5path, "a") as h5:
        size = len(h5["/events/deform"])
        h5.attrs["setup:software version"] = "dclab 0.48.0"
        h5[f"/events/{feat}"] = np.ones(size)

    tmp_path = pathlib.Path(tempfile.mkdtemp("test_212")) / "test_212.rtdc"

    with new_dataset(h5path) as ds:
        # "feat" is defective
        assert feat not in ds.features_innate
        ds.export.hdf5(tmp_path, features=["deform", feat])

    with new_dataset(tmp_path) as ds:
        assert feat in ds.features_innate
        assert not np.all(ds[feat] == 1)


def test_defective_feature_inert_ratio_control_1():
    feat = "inert_ratio_prnc"
    # see https://github.com/DC-analysis/dclab/issues/212
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    # sanity check
    with h5py.File(h5path, "a") as h5:
        size = len(h5["/events/deform"])
        h5.attrs["setup:software version"] = "dclab 0.48.3"
        h5[f"/events/{feat}"] = np.ones(size)

    with new_dataset(h5path) as ds:
        # "feat" is defective
        assert feat in ds.features_innate
        assert np.all(ds[feat] == 1)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_defective_feature_inert_ratio_control_2():
    feat = "inert_ratio_prnc"
    # see https://github.com/DC-analysis/dclab/issues/212
    h5path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    # sanity check
    with h5py.File(h5path, "a") as h5:
        size = len(h5["/events/deform"])
        h5.attrs["setup:software version"] = "dclab 0.48.0"
        h5[f"/events/{feat}"] = np.ones(size)

    with new_dataset(h5path) as ds:
        # "feat" is defective
        assert feat in ds.features_innate
        assert np.all(ds[feat] == 1)


def test_defective_feature_time_issue_204_float32():
    # see https://github.com/DC-analysis/dclab/issues/204
    h5path = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    # sanity check
    with h5py.File(h5path, "r") as h5:
        assert h5["events/time"].dtype.char == "f"

    with new_dataset(h5path) as ds:
        # "time" is a rapid feature, so it will be in features_innate
        assert "time" not in ds.features_innate


def test_defective_feature_time_issue_204_float32_exported():
    # see https://github.com/DC-analysis/dclab/issues/204
    h5path = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    # sanity check
    with h5py.File(h5path, "r") as h5:
        assert h5["events/time"].dtype.char == "f"

    tmp_path = pathlib.Path(tempfile.mkdtemp("test_204")) / "test_204.rtdc"

    with new_dataset(h5path) as ds:
        # "time" is a rapid feature, so it will be in features_innate
        ds.export.hdf5(tmp_path, features=["deform", "time"])

    with new_dataset(tmp_path) as ds:
        # time should now be innate
        assert "time" in ds.features_innate


def test_defective_feature_time_issue_204_noancil():
    # see https://github.com/DC-analysis/dclab/issues/204
    h5path = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    # sanity check
    with h5py.File(h5path, "a") as h5:
        assert h5["events/time"].dtype.char == "f"
        del h5["events/frame"]

    with new_dataset(h5path) as ds:
        assert "time" in ds.features_innate  # now it's there


def test_defective_feature_time_issue_204_noshapein():
    # see https://github.com/DC-analysis/dclab/issues/204
    h5path = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    # sanity check
    with h5py.File(h5path, "a") as h5:
        assert h5["events/time"].dtype.char == "f"
        time = np.arange(h5["events/time"].size, dtype=float)
        del h5["events/time"]
        h5["events/time"] = time
        # Set the software version to something else than Shape-In, but with
        # a lower version of dclab.
        h5.attrs["setup:software version"] = "Other device | dclab 0.30.0"

    with new_dataset(h5path) as ds:
        assert "time" in ds.features_innate  # It's not Shape-In, thus ignored


def test_defective_feature_time_issue_204_old_dclab():
    # see https://github.com/DC-analysis/dclab/issues/204
    h5path = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    # sanity check
    with h5py.File(h5path, "a") as h5:
        assert h5["events/time"].dtype.char == "f"
        time = np.arange(h5["events/time"].size, dtype=float)
        del h5["events/time"]
        h5["events/time"] = time
        # Set the software version to something else than Shape-In, but with
        # a lower version of dclab.
        h5.attrs["setup:software version"] = "ShapeIn 2.0.2 | dclab 0.30.0"

    with new_dataset(h5path) as ds:
        assert "time" not in ds.features_innate  # Recognized as old dclab


def test_defective_feature_time_issue_204_new_dclab():
    # see https://github.com/DC-analysis/dclab/issues/204
    h5path = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    # sanity check
    with h5py.File(h5path, "a") as h5:
        assert h5["events/time"].dtype.char == "f"
        time = np.arange(h5["events/time"].size, dtype=float)
        del h5["events/time"]
        h5["events/time"] = time
        # Set the software version to something else than Shape-In, but with
        # a lower version of dclab.
        h5.attrs["setup:software version"] = "ShapeIn 2.0.2 | dclab 0.50.0"

    with new_dataset(h5path) as ds:
        assert "time" in ds.features_innate  # Recognized as new dclab


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_defective_feature_time_issue_207_with_offset():
    # see https://github.com/DC-analysis/dclab/issues/207
    h5path = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # sanity check
    with h5py.File(h5path, "a") as h5:
        h5.attrs["imaging:frame rate"] = 100
        # write modified frames
        frame_w_offset = h5["events/frame"][:]
        frame_w_offset += 42 - frame_w_offset[0]
        del h5["events/frame"]
        h5["events/frame"] = frame_w_offset
        # write wrong time
        bad_time = np.arange(h5["events/time"].size, dtype=np.float32)
        del h5["events/time"]
        h5["events/time"] = bad_time

    with new_dataset(h5path) as ds:
        assert "time" not in ds.features_innate  # Recognized as float32
        assert np.allclose(ds["time"][0], 0.42, atol=0, rtol=1e-7)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_defective_feature_volume():
    # see https://github.com/ZELLMECHANIK-DRESDEN/ShapeOut/issues/241
    h5path = retrieve_data("fmt-hdf5_mask-contour_2018.zip")

    with h5py.File(h5path, "r") as h5:
        # This volume was computed with an old version of dclab
        wrong_volume = h5["events/volume"][:]

    with new_dataset(h5path) as ds:
        assert "volume" not in ds.features_innate
        correct_volume = ds["volume"]
        assert not np.allclose(wrong_volume, correct_volume)

    # prevent recomputation via logs
    # (do not use context manager here [sic])
    hw = dclab.RTDCWriter(h5path)
    hw.store_log("dclab_issue_141", ["fixed"])
    hw.h5file.close()
    with new_dataset(h5path) as ds2:
        assert np.allclose(ds2["volume"], wrong_volume)

    # remove the log
    with h5py.File(h5path, "a") as h5:
        del h5["logs/dclab_issue_141"]

    # make sure that worked
    with new_dataset(h5path) as ds:
        assert "volume" not in ds.features_innate
        correct_volume = ds["volume"]
        assert not np.allclose(wrong_volume, correct_volume)

    # prevent recomputation via dclab version string
    with h5py.File(h5path, "a") as h5:
        h5.attrs["setup:software version"] = "ShapeIn 2.0.6 | dclab 0.37.0"
    with new_dataset(h5path) as ds2:
        assert "volume" in ds2.features_innate
        assert np.allclose(ds2["volume"], wrong_volume)

    # reset version string
    with h5py.File(h5path, "a") as h5:
        h5.attrs["setup:software version"] = "ShapeIn 2.0.6 | dclab 0.35.1"
    with new_dataset(h5path) as ds2:
        assert "volume" not in ds2.features_innate
        assert np.allclose(ds2["volume"], correct_volume)

    # use context manager to write version number
    with dclab.RTDCWriter(h5path) as hw:
        pass
    with new_dataset(h5path) as ds2:
        assert "volume" in ds2.features_innate
        assert np.allclose(ds2["volume"], wrong_volume)


def test_discouraged_array_dunder_childndarray():
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip"))
    with pytest.warns(UserWarning, match="It may consume a lot of memory"):
        ds["mask"].__array__()


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_hash():
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2017.zip"))
    assert ds.hash == "2c436daba22d2c7397b74d53d80f8931"
    assert ds.format == "hdf5"


def test_fileio_basic():
    """Basic file IO functionality tests"""
    path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    # Create a file object
    fd = io.BytesIO()
    # create a copy
    with h5py.File(fd, "w") as hio, h5py.File(path) as h5:
        rtdc_copy(src_h5file=h5, dst_h5file=hio)

    ds = dclab.new_dataset(fd)
    assert len(ds) == 10
    assert str(ds).count("RTDC_HDF5")
    assert ds.__repr__().count("mm-hdf5")
    assert ds.__repr__().count("BytesIO")
    assert np.allclose(ds["deform"].mean(),
                       0.03620073944330215454,
                       atol=0, rtol=1e-10)
    assert np.allclose(ds.h5file["/events/deform"].attrs["mean"],
                       0.03620073944330215454,
                       atol=0, rtol=1e-10)
    assert "volume" in ds.features
    assert "volume" not in ds.features_innate
    assert np.allclose(ds["volume"][0], 2898.67525650645,
                       rtol=1e-10, atol=0)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_hdf5_load_non_scalar_data():
    """Loading non-scalar data should return data as h5py.Dataset"""
    ds = new_dataset(retrieve_data("fmt-hdf5_image-bg_2020.zip"))
    assert isinstance(ds["image"], h5py.Dataset)
    assert isinstance(ds["image_bg"], h5py.Dataset)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_hdf5_shape_contour():
    ds = new_dataset(retrieve_data("fmt-hdf5_mask-contour_2018.zip"))
    assert "contour" in ds.features_innate
    assert len(ds["contour"]) == 8
    assert ds["contour"].shape == (8, np.nan, 2)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_hdf5_shape_image():
    ds = new_dataset(retrieve_data("fmt-hdf5_image-bg_2020.zip"))
    assert "image" in ds.features_innate
    assert len(ds["image"]) == 5
    assert ds["image"].shape == (5, 80, 250)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_hdf5_shape_mask():
    ds = new_dataset(retrieve_data("fmt-hdf5_image-bg_2020.zip"))
    assert "mask" in ds.features_innate
    assert len(ds["mask"]) == 5
    assert ds["mask"].shape == (5, 80, 250)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_hdf5_shape_trace():
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
    assert len(ds) == 7
    assert "trace" in ds.features_innate
    assert ds["trace"].shape == (6, 7, 177)
    assert ds["trace"]["fl1_raw"].shape == (7, 177)
    assert ds["trace"]["fl1_raw"][0].shape == (177,)
    assert len(ds["trace"]) == 6
    assert len(ds["trace"]["fl1_raw"]) == 7
    assert len(ds["trace"]["fl1_raw"][0]) == 177


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.skipif(sys.version_info < (3, 9),
                    reason="requires python3.9 or higher")
def test_hdf5_ufuncs():
    path_orig = retrieve_data("fmt-hdf5_fl_2018.zip")
    path_mod = path_orig.with_stem("modified")
    shutil.copy2(path_orig, path_mod)
    with h5py.File(path_mod, "a") as h5:
        h5["events/area_cvx"].attrs["min"] = 10.
        h5["events/area_cvx"].attrs["max"] = 100.
        h5["events/area_cvx"].attrs["mean"] = 12.7

    ds = new_dataset(path_orig)
    ds_mod = new_dataset(path_mod)

    assert len(ds) == 7
    assert len(ds_mod) == 7

    # modified
    assert np.min(ds_mod["area_cvx"]) == 10
    assert np.max(ds_mod["area_cvx"]) == 100
    assert np.mean(ds_mod["area_cvx"]) == 12.7

    # reference
    assert np.min(ds["area_cvx"]) == 226.0
    assert np.max(ds["area_cvx"]) == 287.5
    assert np.allclose(np.mean(ds["area_cvx"]), 255.28572, rtol=0, atol=1e-5)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_ignore_empty_hdf5_meta_data_attribute():
    """Ignore empty hdf5 attributes / dclab metadata"""
    # see https://github.com/DC-analysis/dclab/issues/109
    path = retrieve_data("fmt-hdf5_fl_2018.zip")
    # add empty attribute
    with h5py.File(path, "r+") as h5:
        h5.attrs["setup:module composition"] = ""

    # assert
    with pytest.warns(config.EmptyConfigurationKeyWarning,
                      match=r"Empty value for \[setup\]: 'module composition'!"
                      ):
        new_dataset(path)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_ignore_unknown_hdf5_meta_data_attribute():
    """Ignore unknown hdf5 attributes / dclab metadata"""
    # see https://github.com/DC-analysis/dclab/issues/109
    path = retrieve_data("fmt-hdf5_fl_2018.zip")
    # add empty attribute
    with h5py.File(path, "r+") as h5:
        h5.attrs["setup:shizzle"] = ""

    # assert
    with pytest.warns(config.UnknownConfigurationKeyWarning,
                      match=r"Unknown key 'shizzle' in the 'setup' section!"
                      ):
        new_dataset(path)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_image_basic():
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2017.zip"))
    assert np.allclose(np.average(ds["image"][1]), 125.37133333333334)
    assert len(ds["image"]) == 5


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_image_bg():
    path = retrieve_data("fmt-hdf5_fl_2017.zip")
    # add a fake image_bg column
    with rtdc_dataset.RTDCWriter(path) as hw:
        image_bg = hw.h5file["events"]["image"][:] // 2
        hw.store_feature("image_bg", image_bg)

    with new_dataset(path) as ds:
        for ii in range(len(ds)):
            assert np.all(ds["image"][ii] // 2 == ds["image_bg"][ii])


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_image_bg_2():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    with new_dataset(path) as ds:
        assert "image_bg" in ds
        bgc = ds["image"][0] - ds["image_bg"][0]
        assert bgc[10, 11] == 6


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
@pytest.mark.parametrize("idxs",
                         [slice(0, 5, 2),
                          [0, 2, 4],
                          np.array([0, 2, 4]),
                          np.array([True, False, True, False, True])])
def test_index_slicing_contour(idxs):
    data = retrieve_data("fmt-hdf5_fl_2017.zip")
    ds = new_dataset(data)

    contour_ref = [
        ds["contour"][0],
        ds["contour"][2],
        ds["contour"][4],
    ]

    contour_slice = ds["contour"][idxs]

    assert np.all(contour_slice[0] == contour_ref[0])
    assert np.all(contour_slice[1] == contour_ref[1])
    assert np.all(contour_slice[2] == contour_ref[2])


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_logs():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")

    with new_dataset(path_in) as ds:
        assert not ds.logs

    # write some logs
    with h5py.File(path_in, "a") as h5:
        hw = rtdc_dataset.RTDCWriter(h5)
        hw.store_log("test_log",  ["peter", "hans"])

    with new_dataset(path_in) as ds:
        assert ds.logs
        assert ds.logs["test_log"][0] == "peter"

    # remove logs
    with h5py.File(path_in, "a") as h5:
        del h5["logs"]

    with new_dataset(path_in) as ds:
        assert not ds.logs
        try:
            ds.logs["test_log"]
        except KeyError:  # no log data
            pass


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_no_suffix():
    """Loading an .rtdc file that has a wrong suffix"""
    path = str(retrieve_data("fmt-hdf5_mask-contour_2018.zip"))
    path2 = path + ".wrong_suffix"
    os.rename(path, path2)
    ds = new_dataset(path2)
    assert (len(ds) == 8)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_online_polygon_filters():
    """Shape-In 2.3 supports online polygon filters"""
    path = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # add an artificial online polygon filter
    with h5py.File(path, "a") as h5:
        # set soft filter to True
        h5.attrs["online_filter:area_um,deform soft limit"] = True
        # set filter values
        pf_name = "online_filter:area_um,deform polygon points"
        area_um = h5["events"]["area_um"]
        deform = h5["events"]["deform"]
        pf_points = np.array([
            [np.mean(area_um) + np.std(area_um),
             np.mean(deform)],
            [np.mean(area_um) + np.std(area_um),
             np.mean(deform) + np.std(deform)],
            [np.mean(area_um),
             np.mean(deform) + np.std(deform)],
        ])
        h5.attrs[pf_name] = pf_points

    # see if we can open the file without any error
    with new_dataset(path) as ds:
        assert len(ds) == 8
        assert ds.config["online_filter"]["area_um,deform soft limit"]
        assert "area_um,deform polygon points" in ds.config["online_filter"]
        assert np.allclose(
            ds.config["online_filter"]["area_um,deform polygon points"],
            pf_points)


def test_open_simple_file(tmp_path):
    path = tmp_path / "test.rtdc"
    with h5py.File(path, "w") as h5:
        h5["events/time"] = np.arange(100)
        h5["events/deform"] = np.linspace(0.1, 0.2, 100)

    with dclab.new_dataset(path) as ds:
        assert "software version" not in ds.config["setup"]
        assert ds.title == "undefined sample - M0"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_open_with_invalid_feature_names():
    """Loading an .rtdc file that has wrong feature name"""
    path = str(retrieve_data("fmt-hdf5_mask-contour_2018.zip"))
    # add wrong feature name right at the top of the list
    with h5py.File(path, "r+") as h5:
        h5["events"]["a0"] = h5["events"]["deform"]
    # see if we can open the file without any error
    ds = new_dataset(path)
    assert len(ds) == 8


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_trace():
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2017.zip"))
    assert len(ds["trace"]) == 2
    assert ds["trace"]["fl1_raw"].shape == (5, 100)
    assert np.allclose(np.average(
        ds["trace"]["fl1_median"][0]), 0.027744706519425219)
