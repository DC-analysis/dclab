#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test tdms file format"""
from __future__ import print_function

import os
import pathlib
import shutil
import sys
import tempfile

import numpy as np
import pytest

from dclab import new_dataset
import dclab.rtdc_dataset.fmt_tdms.naming

from helper_methods import retrieve_data, example_data_sets, cleanup


def test_compatibility_minimal():
    ds = new_dataset(retrieve_data("rtdc_data_minimal.zip"))
    assert ds.config["setup"]["channel width"] == 20
    assert ds.config["setup"]["chip region"].lower() == "channel"
    assert ds.config["setup"]["flow rate"] == 0.12
    assert ds.config["imaging"]["pixel size"] == 0.34
    cleanup()


@pytest.mark.skipif(sys.version_info < (3, 6),
                    reason="requires python3.6 or higher")
def test_compatibility_channel_width():
    # At some point, "Channel width" was repleaced by "Channel width [um]"
    path = retrieve_data("rtdc_data_minimal.zip")
    para = path.parent / "M1_para.ini"
    pardata = para.read_text()
    pardata = pardata.replace("Channel width = 20\n", "Channel width = 34\n")
    para.write_text(pardata)
    ds = new_dataset(path)
    assert ds.config["setup"]["channel width"] == 34
    cleanup()


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.event_contour.'
                            + 'NoContourDataWarning')
def test_compatibility_shapein201():
    ds = new_dataset(retrieve_data("rtdc_data_shapein_v2.0.1.zip"))
    assert ds.config["setup"]["channel width"] == 20
    assert ds.config["setup"]["chip region"].lower() == "channel"
    assert ds.config["setup"]["software version"] == "ShapeIn 2.0.1"
    assert ds.config["imaging"]["pixel size"] == 0.34
    assert ds.config["imaging"]["flash duration"] == 2
    assert ds.config["experiment"]["date"] == "2017-10-12"
    assert ds.config["experiment"]["time"] == "12:54:31"
    cleanup()


def test_contour_basic():
    ds = new_dataset(retrieve_data(example_data_sets[1]))
    assert len(ds["contour"]) == 12
    assert np.allclose(np.average(ds["contour"][0]), 38.488764044943821)
    assert ds["contour"]._initialized
    cleanup()


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.event_contour.'
                            + 'NoContourDataWarning')
def test_contour_naming():
    # Test that we always find the correct contour name
    ds = new_dataset(retrieve_data(example_data_sets[0]))
    dp = pathlib.Path(ds.path).resolve()
    dn = dp.parent
    contfile = dn / "M1_0.120000ul_s_contours.txt"
    contfileshort = dn / "M1_contours.txt"
    contfileexact = dn / "M1_2us_70A_0.120000ul_s_contours.txt"
    del ds

    # Test for perfect match
    # "M1_2us_70A_0.120000ul_s_contours.txt" should have priority over
    # "M1_contours.txt" and "M1_0.120000ul_s_contours.txt".
    shutil.copy(str(contfile), str(contfileshort))
    shutil.copy(str(contfile), str(contfileexact))
    ds2 = new_dataset(dp)
    assert str(ds2["contour"].identifier) == str(contfileexact)
    assert not np.allclose(ds2["contour"][1], 0)
    del ds2

    # Check if "M1_contours.txt" is used if the other is not
    # there.
    contfileshort.unlink()
    contfileexact.unlink()
    contfile.rename(contfileshort)
    ds3 = new_dataset(dp)
    assert str(ds3["contour"].identifier) == str(contfileshort)
    del ds3
    contfileshort.rename(contfile)

    # Create M10 file
    with (dn / "M10_contours.txt").open(mode="w"):
        pass
    ds4 = new_dataset(dp)
    assert str(ds4["contour"].identifier) == str(contfile)
    del ds4

    # Check when there is no contour file
    os.remove(str(contfile))
    # This will issue a warning that no contour data was found.
    ds5 = new_dataset(dp)
    assert ds5["contour"].identifier is None


def test_contour_negative_offset():
    ds = new_dataset(retrieve_data(example_data_sets[1]))
    ds["contour"][0]
    ds["contour"].event_offset = 1
    assert np.all(ds["contour"][0] == np.zeros((2, 2), dtype=int))
    cleanup()


def test_contour_not_initialized():
    ds = new_dataset(retrieve_data(example_data_sets[1]))
    assert not ds["contour"]._initialized
    cleanup()


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.event_contour.'
                            + 'NoContourDataWarning')
def test_fluorescence_config():
    ds1 = new_dataset(retrieve_data("rtdc_data_minimal.zip"))
    assert "fluorescence" not in ds1.config
    ds2 = new_dataset(retrieve_data("rtdc_data_traces_2flchan.zip"))
    assert "fluorescence" in ds2.config
    cleanup()


def test_image_basic():
    ds = new_dataset(retrieve_data(example_data_sets[1]))
    # Transition image
    assert np.allclose(ds["image"][0], 0)
    # Real image
    assert np.allclose(np.average(ds["image"][1]), 45.1490478515625)
    cleanup()


def test_image_column_length():
    ds = new_dataset(retrieve_data(example_data_sets[1]))
    assert len(ds["image"]) == 3
    cleanup()


def test_image_out_of_bounds():
    ds = new_dataset(retrieve_data(example_data_sets[1]))
    try:
        ds["image"][5]
    except IndexError:
        pass
    else:
        raise ValueError("IndexError should have been raised!")
    cleanup()


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'ancillaries.ancillary_feature.'
                            + 'BadFeatureSizeWarning')
def test_large_fov():
    ds = new_dataset(retrieve_data(example_data_sets[3]))
    # initial image is missing
    assert np.allclose(ds["image"][0], 0)
    # initial contour is empty
    assert np.allclose(ds["contour"][0], 0)
    # maximum of contour is larger than 255 (issue #167)
    assert ds["contour"][1].max() == 815
    # compute brightness with given contour
    # Remove the brightness column and let it recompute
    # using the ancillary columns. Besides testing the
    # correct positioning of the contour, this is a
    # sanity test for the brightness computation.
    bavg = ds._events.pop("bright_avg")
    bcom = ds["bright_avg"]
    assert np.allclose(bavg[1], bcom[1])
    cleanup()


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.event_contour.'
                            + 'NoContourDataWarning')
def test_load_tdms_all():
    for ds in example_data_sets:
        tdms_path = retrieve_data(ds)
        ds = new_dataset(tdms_path)
    cleanup()


def test_load_tdms_avi_files():
    tdms_path = retrieve_data(example_data_sets[1])
    edest = pathlib.Path(tdms_path).parent
    ds1 = new_dataset(tdms_path)
    assert pathlib.Path(ds1["image"].video_file).name == "M1_imaq.avi"
    shutil.copyfile(str(edest / "M1_imaq.avi"),
                    str(edest / "M1_imag.avi"))
    ds2 = new_dataset(tdms_path)
    # prefer imag over imaq
    assert pathlib.Path(ds2["image"].video_file).name == "M1_imag.avi"
    shutil.copyfile(str(edest / "M1_imaq.avi"),
                    str(edest / "M1_test.avi"))
    ds3 = new_dataset(tdms_path)
    # ignore any other videos
    assert pathlib.Path(ds3["image"].video_file).name == "M1_imag.avi"
    os.remove(str(edest / "M1_imaq.avi"))
    os.remove(str(edest / "M1_imag.avi"))
    ds4 = new_dataset(tdms_path)
    # use available video if ima* not there
    assert pathlib.Path(ds4["image"].video_file).name == "M1_test.avi"
    cleanup()


def test_load_tdms_simple():
    tdms_path = retrieve_data(example_data_sets[0])
    ds = new_dataset(tdms_path)
    assert ds._filter.shape[0] == 156
    cleanup()


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'ancillaries.ancillary_feature.'
                            + 'BadFeatureSizeWarning')
def test_mask_basic():
    ds = new_dataset(retrieve_data(example_data_sets[1]))
    assert len(ds["mask"]) == 12
    # Test mask computation by averaging brightness and comparing to
    # the ancillary feature "bright_avg".
    bavg1 = ds["bright_avg"][1]
    bavg2 = np.mean(ds["image"][1][ds["mask"][1]])
    assert np.allclose(bavg1, bavg2), "mask is correctly computed from contour"
    cleanup()


def test_mask_img_shape():
    ds = new_dataset(retrieve_data(example_data_sets[1]))
    # shape from configuration
    assert ds["mask"]._img_shape == (96, 256)
    # shape from image data
    ds.config["imaging"].pop("roi size x")
    ds.config["imaging"].pop("roi size y")
    ds["mask"]._shape = None
    assert ds["mask"]._img_shape == (96, 256)
    # no shape available
    ds._events.pop("image")
    ds["mask"].image = None
    ds["mask"]._shape = None
    assert ds["mask"]._img_shape == (0, 0)
    assert len(ds["mask"]) == 0


def test_naming_valud():
    for key in dclab.rtdc_dataset.fmt_tdms.naming.dclab2tdms:
        assert key in dclab.definitions.feature_names


def test_pixel_size():
    path = retrieve_data("rtdc_data_minimal.zip")
    para = path.parent / "M1_para.ini"
    data = para.open("r").read()
    newdata = data.replace("Pix Size = 0.340000", "Pix Size = 0.120000")
    with para.open("w") as fd:
        fd.write(newdata)
    ds = new_dataset(path)
    assert ds.config["imaging"]["pixel size"] == 0.12
    cleanup()


def test_project_path():
    tfile = retrieve_data(example_data_sets[0])
    ds = dclab.new_dataset(tfile)
    assert ds.hash == "69733e31b005c145997fac8a22107ded"
    assert ds.format == "tdms"
    tpath = pathlib.Path(tfile).resolve()
    a = dclab.rtdc_dataset.fmt_tdms.get_project_name_from_path(str(tpath))
    b = dclab.rtdc_dataset.fmt_tdms.get_project_name_from_path(
        str(tpath.parent))
    assert a == b
    c = dclab.rtdc_dataset.fmt_tdms.get_project_name_from_path(
        str(tpath.parent / "online" / tpath.name))
    d = dclab.rtdc_dataset.fmt_tdms.get_project_name_from_path(
        str(tpath.parent / "online" / "data" / tpath.name))
    e = dclab.rtdc_dataset.fmt_tdms.get_project_name_from_path(
        str(tpath.parent / "online" / "data"))

    assert a == e
    assert a == c
    assert a == d
    cleanup()


def test_trace_basic():
    ds = new_dataset(retrieve_data(example_data_sets[1]))
    msg = "traces should not be loaded into memory before first access"
    assert ds["trace"].__repr__().count("<not loaded into memory>"), msg
    assert len(ds["trace"]) == 2
    assert np.allclose(np.average(
        ds["trace"]["fl1_median"][0]), 287.08999999999997)
    cleanup()


def test_trace_import_fail():
    # make sure undefined trace data does not raise an error
    tdms_path = retrieve_data(example_data_sets[1])
    dclab.definitions.FLUOR_TRACES.append("peter")
    dclab.rtdc_dataset.fmt_tdms.naming.tr_data_map["peter"] = [u'ukwn', u'ha']
    new_dataset(tdms_path)
    # clean up
    dclab.rtdc_dataset.fmt_tdms.naming.tr_data_map.pop("peter")
    dclab.definitions.FLUOR_TRACES.pop(-1)
    cleanup()


def test_trace_methods():
    ds = new_dataset(retrieve_data(example_data_sets[1]))
    for k in list(ds["trace"].keys()):
        assert k in dclab.definitions.FLUOR_TRACES
    for k in ds["trace"]:
        assert k in dclab.definitions.FLUOR_TRACES
    assert ds["trace"].__repr__().count("<loaded into memory>")
    cleanup()


def test_unicode_paths():
    path = retrieve_data(example_data_sets[1])
    path = pathlib.Path(path)
    pp = path.parent
    # create a unicode name
    pp2 = pathlib.Path(tempfile.mktemp(prefix="dclàb_tést_asgård_únícodè"))
    pp.rename(pp2)
    ds = new_dataset(pp2 / path.name)
    ds.__repr__()
    shutil.rmtree(str(pp2), ignore_errors=True)
    cleanup()


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
