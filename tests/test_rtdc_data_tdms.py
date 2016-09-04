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
from dclab import RTDC_DataSet

from helper_methods import example_data_dict, retreive_tdms, example_data_sets


def test_load_tdms_simple():
    tdms_path = retreive_tdms(example_data_sets[0])
    ds = RTDC_DataSet(tdms_path)
    assert ds._filter.shape[0] == 156
    

if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
