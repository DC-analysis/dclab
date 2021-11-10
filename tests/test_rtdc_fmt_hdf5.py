"""Test hdf5 file format"""
import os

import h5py
import numpy as np
import pytest

import dclab
from dclab import new_dataset, rtdc_dataset
from dclab.rtdc_dataset import config

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


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_hash():
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2017.zip"))
    assert ds.hash == "2c436daba22d2c7397b74d53d80f8931"
    assert ds.format == "hdf5"


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
def test_ignore_empty_hdf5_meta_data_attribute():
    """Ignore empty hdf5 attributes / dclab metadata"""
    # see https://github.com/ZELLMECHANIK-DRESDEN/dclab/issues/109
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
    # see https://github.com/ZELLMECHANIK-DRESDEN/dclab/issues/109
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
    assert(len(ds) == 8)


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


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in sorted(list(loc.keys())):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
