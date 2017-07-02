#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Downsampling only affects RTDCBase._plot_filter
"""
from __future__ import print_function

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
import dclab

from helper_methods import example_data_dict



def test_limit_simple():
    keys = ["area_um", "deform", "time", "frame"]
    ddict = example_data_dict(size=9999, keys=keys)
    ds = dclab.new_dataset(ddict)

    assert np.sum(ds._filter) == 9999
    filtflt = {"limit events": 800}
    
    cfg = {"filtering": filtflt}
    ds.config.update(cfg)
    ds.apply_filter()
    
    assert np.sum(ds._filter) == 800


def test_limit_equal():
    keys = ["area_um", "deform", "time", "frame"]
    ddict = example_data_dict(size=9999, keys=keys)
    ds = dclab.new_dataset(ddict)

    assert np.sum(ds._filter) == 9999
    filtflt = {"limit events": 9999}
    
    cfg = {"filtering": filtflt}
    ds.config.update(cfg)
    ds.apply_filter()
    
    assert np.sum(ds._filter) == 9999


def test_limit_above():
    keys = ["area_um", "deform", "time", "frame"]
    ddict = example_data_dict(size=9999, keys=keys)
    ds = dclab.new_dataset(ddict)

    assert np.sum(ds._filter) == 9999
    filtflt = {"limit events": 10000}
    
    cfg = {"filtering": filtflt}
    ds.config.update(cfg)
    ds.apply_filter()
    
    assert np.sum(ds._filter) == 9999


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
