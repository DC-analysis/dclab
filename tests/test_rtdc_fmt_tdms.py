#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test tdms file format"""
from __future__ import print_function

import io
import os
from os.path import abspath, dirname, join, basename
import shutil
import sys
import tempfile
import warnings
import zipfile

import numpy as np


# Add parent directory to beginning of path variable
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from dclab import new_dataset
import dclab.rtdc_dataset.fmt_tdms.naming

from helper_methods import example_data_dict, retreive_tdms, example_data_sets, cleanup


def test_contour_basic():
    ds = new_dataset(retreive_tdms(example_data_sets[1]))
    assert len(ds["contour"]) == 12
    assert np.allclose(np.average(ds["contour"][0]), 38.488764044943821)
    assert ds["contour"]._initialized
    cleanup()


def test_contour_naming():
    # Test that we always find the correct contour name
    ds = new_dataset(retreive_tdms(example_data_sets[0]))
    dp = ds.path
    dn = dirname(dp)
    contfile = join(dn, "M1_0.120000ul_s_contours.txt")
    contfileshort = join(dn, "M1_contours.txt")
    del ds
    
    # "M1_0.120000ul_s_contours.txt" should have priority over
    # "M1_contours.txt".
    with io.open(contfileshort, "w") as _fd:
        pass
    ds2 = new_dataset(dp)
    assert ds2["contour"].identifier == contfile
    assert not np.allclose(ds2["contour"][1], 0)
    del ds2
    
    # Check if "M1_contours.txt" is used if the other is not
    # there.
    os.remove(contfileshort)
    os.rename(contfile, contfileshort)
    ds3 = new_dataset(dp)
    assert ds3["contour"].identifier == contfileshort
    del ds3
    os.rename(contfileshort, contfile)
    
    # Create M10 file
    with io.open(join(dn, "M10_contours.txt"), "w") as _fd:
        pass
    ds4 = new_dataset(dp)
    assert ds4["contour"].identifier == contfile
    del ds4
    
    # Check when there is no contour file
    os.remove(contfile)
    # This will issue a warning that no contour data was found.
    ds5 = new_dataset(dp)
    assert ds5["contour"].identifier is None


def test_contour_negative_offset():
    ds = new_dataset(retreive_tdms(example_data_sets[1]))
    _a = ds["contour"][0]
    ds["contour"].event_offset = 1
    assert np.all(ds["contour"][0] == np.zeros((2,2), dtype=int))
    cleanup()


def test_contour_not_initialized():
    ds = new_dataset(retreive_tdms(example_data_sets[1]))
    assert ds["contour"]._initialized == False
    cleanup()


def test_image_basic():
    ds = new_dataset(retreive_tdms(example_data_sets[1]))
    # Transition image
    assert np.all(np.isnan(ds["image"][0]))
    # Real image
    assert np.allclose(np.average(ds["image"][1]), 45.1490478515625)
    cleanup()


def test_image_column_length():
    ds = new_dataset(retreive_tdms(example_data_sets[1]))
    assert len(ds["image"]) == 3
    cleanup()


def test_image_out_of_bounds():
    ds = new_dataset(retreive_tdms(example_data_sets[1]))
    try:
        _a = ds["image"][5]
    except IndexError:
        pass
    else:
        raise ValueError("IndexError should have been raised!")
    cleanup()


def test_large_fov():
    ds = new_dataset(retreive_tdms(example_data_sets[3]))
    # initial image is missing
    assert np.all(np.isnan(ds["image"][0]))
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


def test_load_tdms_all():
    for ds in example_data_sets:
        tdms_path = retreive_tdms(ds)
        ds = new_dataset(tdms_path)
    cleanup()


def test_load_tdms_avi_files():
    tdms_path = retreive_tdms(example_data_sets[1])
    edest = dirname(tdms_path)
    ds1 = new_dataset(tdms_path)
    assert os.path.basename(ds1["image"].video_file) == "M1_imaq.avi"
    shutil.copyfile(join(edest, "M1_imaq.avi"),
                    join(edest, "M1_imag.avi"))
    ds2 = new_dataset(tdms_path)
    # prefer imag over imaq
    assert os.path.basename(ds2["image"].video_file) == "M1_imag.avi"
    shutil.copyfile(join(edest, "M1_imaq.avi"),
                    join(edest, "M1_test.avi"))
    ds3 = new_dataset(tdms_path)
    # ignore any other videos
    assert os.path.basename(ds3["image"].video_file) == "M1_imag.avi"
    os.remove(join(edest, "M1_imaq.avi"))
    os.remove(join(edest, "M1_imag.avi"))
    ds4 = new_dataset(tdms_path)
    # use available video if ima* not there
    assert os.path.basename(ds4["image"].video_file) == "M1_test.avi"
    cleanup()


def test_load_tdms_simple():
    tdms_path = retreive_tdms(example_data_sets[0])
    ds = new_dataset(tdms_path)
    assert ds._filter.shape[0] == 156
    cleanup()


def test_trace_basic():
    ds = new_dataset(retreive_tdms(example_data_sets[1]))
    assert ds["trace"].__repr__().count("<not loaded into memory>"), "traces should not be loaded into memory before first access"
    assert len(ds["trace"]) == 6
    assert np.allclose(np.average(ds["trace"]["FL1med"][0]), 287.08999999999997)
    cleanup()


def test_project_path():
    tfile = retreive_tdms(example_data_sets[0])
    ds = dclab.new_dataset(tfile)
    assert ds.hash == "69733e31b005c145997fac8a22107ded"
    assert ds.format == "tdms"
    a = dclab.rtdc_dataset.fmt_tdms.get_project_name_from_path(tfile)
    b = dclab.rtdc_dataset.fmt_tdms.get_project_name_from_path(dirname(tfile))
    assert a == b
    c = dclab.rtdc_dataset.fmt_tdms.get_project_name_from_path(dirname(tfile)+"/online/"+basename(tfile))
    d = dclab.rtdc_dataset.fmt_tdms.get_project_name_from_path(dirname(tfile)+"/online/data/"+basename(tfile))
    e = dclab.rtdc_dataset.fmt_tdms.get_project_name_from_path(dirname(tfile)+"/online/data/")
    
    assert a == e
    assert a == c
    assert a == d 
    cleanup()


def test_trace_import_fail():
    tdms_path = retreive_tdms(example_data_sets[1])
    dclab.rtdc_dataset.fmt_tdms.naming.tr_data.append([u'fluorescence traces', u'peter'])
    _ds1 = new_dataset(tdms_path)
    # clean up
    dclab.rtdc_dataset.fmt_tdms.naming.tr_data.pop(-1)
    cleanup()


def test_trace_methods():
    ds = new_dataset(retreive_tdms(example_data_sets[1]))
    for k in list(ds["trace"].keys()):
        assert  k in  [u'FL1med', u'FL2raw', u'FL2med', u'FL3med', u'FL1raw', u'FL3raw']
    for k in ds["trace"]:
        assert  k in  [u'FL1med', u'FL2raw', u'FL2med', u'FL3med', u'FL1raw', u'FL3raw']
    assert ds["trace"].__repr__().count("<loaded into memory>")
    cleanup()


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
