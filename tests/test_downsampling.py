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
import dclab

from helper_methods import example_data_dict


def test_downsample_none():
    keys = ["Area", "Defo", "Time", "Frame", "FL-3width"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.RTDC_DataSet(ddict=ddict)

    assert np.sum(ds._plot_filter) == 8472
    
    pltfilt = {"Downsample Events": 100,
               "Downsampling": False}

    cfg = {"Plotting": pltfilt}
    ds.UpdateConfiguration(cfg)
    ds.ApplyFilter()
    ds.GetDownSampledScatter()
    assert np.sum(ds._plot_filter) == 8472

    # Do it again with kde vals
    ds.GetDownSampledScatter(c=np.arange(8472))
    


def test_downsample_none2():
    keys = ["Area", "Defo", "Time", "Frame", "FL-3width"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.RTDC_DataSet(ddict=ddict)

    assert np.sum(ds._plot_filter) == 8472
    
    pltfilt = {"Downsample Events": 100,
               "Downsampling": True}
    filtflt = {"Enable Filters": False}
    
    cfg = {"Plotting": pltfilt,
           "Filtering": filtflt}
    ds.UpdateConfiguration(cfg)
    ds.ApplyFilter()
    ds.GetDownSampledScatter()
    
    assert np.sum(ds._plot_filter) == 100
    assert np.sum(ds._filter) == 8472

    filtflt["Enable Filters"] = True
    ds.UpdateConfiguration(cfg)
    ds.ApplyFilter()
    ds.GetDownSampledScatter()

    assert np.sum(ds._plot_filter) == 100
    assert np.sum(ds._filter) == 8472



def test_downsample_yes():
    """ Simple downsampling test.
    """
    keys = ["Area", "Defo", "Time", "Frame", "FL-3width"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.RTDC_DataSet(ddict=ddict)

    assert np.sum(ds._plot_filter) == 8472
    
    pltfilt = {"Downsample Events": 100,
               "Downsampling": True}
    
    cfg = {"Plotting": pltfilt}
    ds.UpdateConfiguration(cfg)
    ds.ApplyFilter()
    ds.GetDownSampledScatter()
    assert np.sum(ds._plot_filter) == 100
    ds.GetDownSampledScatter()
    
    # Do it again with kde vals
    ds.GetDownSampledScatter(c=np.arange(8472))
    
    

def test_downsample_up():
    """
    Likely causes removal of too many points and requires
    re-inserting them.
    """
    keys = ["Area", "Defo", "Time", "Frame", "FL-3width"]
    ddict = example_data_dict(size=10000, keys=keys)
    ds = dclab.RTDC_DataSet(ddict=ddict)

    assert np.sum(ds._plot_filter) == 10000
    
    pltfilt = {"Downsample Events":9999,
               "Downsampling": True}
    
    cfg = {"Plotting": pltfilt}
    ds.UpdateConfiguration(cfg)
    ds.ApplyFilter()
    ds.GetDownSampledScatter()
    assert np.sum(ds._plot_filter) == 9999
    ds.GetDownSampledScatter()





if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
