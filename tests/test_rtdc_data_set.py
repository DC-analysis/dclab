#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Downsampling only affects RTDC_DataSet._plot_filter
"""
from __future__ import print_function

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


def test_wrong_things():
    ddict = example_data_dict(size=67, keys=["Area", "Defo"])
    ds = RTDC_DataSet(ddict=ddict)
    
    # Check unknown variable (warning will be displayed
    try:
        ds.ApplyFilter(force=["on_purpose_unknown"])
    except ValueError:
        pass


def test_min_max_update():
    ddict = example_data_dict(size=67, keys=["area", "defo"])
    ds = RTDC_DataSet(ddict=ddict)
    cfg = ds.config.copy()

    # Force updating circularity
    cfg["filtering"]["defo min"] = .4
    cfg["filtering"]["defo max"] = .8
    ds.config.update(cfg)

    ds.ApplyFilter()


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
