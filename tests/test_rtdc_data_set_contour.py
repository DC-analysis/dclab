#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Test functions for loading contours
"""
from __future__ import print_function, unicode_literals

import codecs
import copy
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
from dclab import RTDC_DataSet

from helper_methods import example_data_dict, retreive_tdms, example_data_sets


def test_contour_basic():
    ds = RTDC_DataSet(tdms_path = retreive_tdms(example_data_sets[1]))
    assert len(ds.contour) == 12
    assert np.allclose(np.average(ds.contour[0]), 38.488764044943821)
    assert ds.contour._initialized


def test_contour_negative_offset():
    ds = RTDC_DataSet(tdms_path = retreive_tdms(example_data_sets[1]))
    _a = ds.contour[0]
    ds.contour.event_offset = 1
    assert np.all(ds.contour[0] == np.zeros((2,2), dtype=int))


def test_contour_not_initialized():
    ds = RTDC_DataSet(tdms_path = retreive_tdms(example_data_sets[1]))
    assert ds.contour._initialized == False


def test_contour_not_available():
    ddict = example_data_dict(size=67, keys=["Area", "Defo"])
    ds = RTDC_DataSet(ddict=ddict)    
    try:
        a = ds.contour[0]
    except IndexError:
        pass
    else:
        raise ValueError("Index error should have been raised!")



if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
