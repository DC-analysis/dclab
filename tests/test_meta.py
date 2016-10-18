#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Downsampling only affects RTDC_DataSet._plot_filter
"""
from __future__ import print_function

import codecs
import copy
import numpy as np
import os
from os.path import abspath, dirname, join, basename
import shutil
import sys
import tempfile
import warnings
import zipfile


# Add parent directory to beginning of path variable
sys.path.insert(0, dirname(dirname(abspath(__file__))))
import dclab
from dclab import RTDC_DataSet

from helper_methods import example_data_dict, retreive_tdms, example_data_sets


def test_project_path():
    tfile = retreive_tdms(example_data_sets[0])
    ds = RTDC_DataSet(tdms_path=tfile)
    assert len(ds.file_hashes) != 0
    a = dclab.rtdc_dataset.GetProjectNameFromPath(tfile)
    b = dclab.rtdc_dataset.GetProjectNameFromPath(dirname(tfile))
    assert a == b
    c = dclab.rtdc_dataset.GetProjectNameFromPath(dirname(tfile)+"/online/"+basename(tfile))
    d = dclab.rtdc_dataset.GetProjectNameFromPath(dirname(tfile)+"/online/data/"+basename(tfile))
    e = dclab.rtdc_dataset.GetProjectNameFromPath(dirname(tfile)+"/online/data/")
    
    assert a == e
    assert a == c
    assert a == d 

if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
