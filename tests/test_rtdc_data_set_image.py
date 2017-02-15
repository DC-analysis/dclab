#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Test functions for loading images
"""
from __future__ import print_function, unicode_literals

import codecs
import copy
import numpy as np
import os
from os.path import abspath, dirname, join
import pytest
import shutil
import sys
import tempfile
import warnings
import zipfile


# Add parent directory to beginning of path variable
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from dclab import RTDC_DataSet

from helper_methods import example_data_dict, retreive_tdms, example_data_sets


TRAVIS = "TRAVIS" in os.environ and os.environ["TRAVIS"].lower() == "true"


@pytest.mark.xfail(TRAVIS, reason="OpenCV install problems")
def test_image_basic():
    ds = RTDC_DataSet(tdms_path = retreive_tdms(example_data_sets[1]))
    # Transition image
    assert np.allclose(np.average(ds.image[0]), 127.03125)
    assert np.allclose(np.average(ds.image[1]), 45.512017144097221)


@pytest.mark.xfail(TRAVIS, reason="OpenCV install problems")
def test_image_column_length():
    ds = RTDC_DataSet(tdms_path = retreive_tdms(example_data_sets[1]))
    assert len(ds.image) == 3


def test_image_not_available():
    ddict = example_data_dict(size=67, keys=["Area", "Defo"])
    ds = RTDC_DataSet(ddict=ddict)    
    try:
        a = ds.image[0]
    except IndexError:
        pass
    else:
        raise ValueError("Index error should have been raised!")


def test_image_out_of_bounds():
    ds = RTDC_DataSet(tdms_path = retreive_tdms(example_data_sets[1]))
    try:
        a = ds.image[5]
    except OSError :
        pass
    else:
        raise ValueError("OS error should have been raised!")



if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
