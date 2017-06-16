#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Downsampling only affects RTDC_DataSet._plot_filter
"""
from __future__ import print_function

import codecs
import numpy as np
import os
from os.path import abspath, dirname, join
import shutil
import sys
import tempfile
import warnings
import zipfile


# Add parent directory to beginning of path variable
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from dclab import RTDC_DataSet, dfn

from helper_methods import example_data_dict, retreive_tdms, example_data_sets



def test_contour_basic():
    ds = RTDC_DataSet(tdms_path = retreive_tdms(example_data_sets[1]))
    assert len(ds["contour"]) == 12
    assert np.allclose(np.average(ds["contour"][0]), 38.488764044943821)
    assert ds["contour"]._initialized


def test_contour_negative_offset():
    ds = RTDC_DataSet(tdms_path = retreive_tdms(example_data_sets[1]))
    _a = ds["contour"][0]
    ds["contour"].event_offset = 1
    assert np.all(ds["contour"][0] == np.zeros((2,2), dtype=int))


def test_contour_not_initialized():
    ds = RTDC_DataSet(tdms_path = retreive_tdms(example_data_sets[1]))
    assert ds["contour"]._initialized == False


def test_image_basic():
    ds = RTDC_DataSet(tdms_path = retreive_tdms(example_data_sets[1]))
    # Transition image
    assert np.allclose(np.average(ds["image"][0]), 127.03125)
    assert np.allclose(np.average(ds["image"][1]), 45.512017144097221)


def test_image_column_length():
    ds = RTDC_DataSet(tdms_path = retreive_tdms(example_data_sets[1]))
    assert len(ds["image"]) == 3


def test_image_out_of_bounds():
    ds = RTDC_DataSet(tdms_path = retreive_tdms(example_data_sets[1]))
    try:
        a = ds["image"][5]
    except IndexError:
        pass
    else:
        raise ValueError("IndexError should have been raised!")


def test_load_tdms_all():
    for ds in example_data_sets:
        tdms_path = retreive_tdms(ds)
        ds = RTDC_DataSet(tdms_path)


def test_load_tdms_avi_files():
    tdms_path = retreive_tdms(example_data_sets[1])
    edest = dirname(tdms_path)
    ds1 = RTDC_DataSet(tdms_path)
    assert os.path.basename(ds1["image"].video_file) == "M1_imaq.avi"
    shutil.copyfile(join(edest, "M1_imaq.avi"),
                    join(edest, "M1_imag.avi"))
    ds2 = RTDC_DataSet(tdms_path)
    # prefer imag over imaq
    assert os.path.basename(ds2["image"].video_file) == "M1_imag.avi"
    shutil.copyfile(join(edest, "M1_imaq.avi"),
                    join(edest, "M1_test.avi"))
    ds3 = RTDC_DataSet(tdms_path)
    # ignore any other videos
    assert os.path.basename(ds3["image"].video_file) == "M1_imag.avi"
    os.remove(join(edest, "M1_imaq.avi"))
    os.remove(join(edest, "M1_imag.avi"))
    ds4 = RTDC_DataSet(tdms_path)
    # use available video if ima* not there
    assert os.path.basename(ds4["image"].video_file) == "M1_test.avi"


def test_load_tdms_simple():
    tdms_path = retreive_tdms(example_data_sets[0])
    ds = RTDC_DataSet(tdms_path)
    assert ds._filter.shape[0] == 156


def test_trace_basic():
    ds = RTDC_DataSet(tdms_path = retreive_tdms(example_data_sets[1]))
    assert ds["trace"].__repr__().count("<not loaded into memory>"), "traces should not be loaded into memory before first access"
    assert len(ds["trace"]) == 6
    assert np.allclose(np.average(ds["trace"]["FL1med"][0]), 287.08999999999997)


def test_trace_import_fail():
    tdms_path = retreive_tdms(example_data_sets[1])
    edest = dirname(tdms_path)
    dfn.tr_data.append([u'fluorescence traces', u'peter'])
    ds1 = RTDC_DataSet(tdms_path)
    # clean up
    dfn.tr_data.pop(-1)


def test_trace_methods():
    ds = RTDC_DataSet(tdms_path = retreive_tdms(example_data_sets[1]))
    for k in list(ds["trace"].keys()):
        assert  k in  [u'FL1med', u'FL2raw', u'FL2med', u'FL3med', u'FL1raw', u'FL3raw']
    for k in ds["trace"]:
        assert  k in  [u'FL1med', u'FL2raw', u'FL2med', u'FL3med', u'FL1raw', u'FL3raw']
    assert ds["trace"].__repr__().count("<loaded into memory>")


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
