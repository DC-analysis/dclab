#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
from os.path import abspath, dirname, join

import numpy as np

# Add parent directory to beginning of path variable
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import dclab

import os


import warnings
import zipfile

from helper_methods import example_data_dict

def test_kde_general():
    ## Download and extract data
    ddict = example_data_dict()
    ds = dclab.RTDC_DataSet(ddict=ddict)
    
    kdes = dclab.config.get_config_entry_choices("Plotting", "KDE")
    
    for kde in kdes:
        ds.Configuration["Plotting"]["KDE"] = kde
        ds.GetKDE_Contour()
        ds.GetKDE_Scatter()


def test_kde_none():
    ddict = example_data_dict()
    ds = dclab.RTDC_DataSet(ddict=ddict)
    
    ds.Configuration["Plotting"]["KDE"] = "None"
    sc = ds.GetKDE_Scatter()
    assert np.sum(sc) == sc.shape[0]

    ds.GetKDE_Contour()


def test_kde_nofilt():
    ddict = example_data_dict()
    ds = dclab.RTDC_DataSet(ddict=ddict)
    
    ds.Configuration["Filtering"]["Enable Filters"] = False
    sc = ds.GetKDE_Scatter()
    cc = ds.GetKDE_Contour()
    assert sc.shape[0] == sc.shape[0]


def test_kde_positions():
    ddict = example_data_dict()
    ds = dclab.RTDC_DataSet(ddict=ddict)
    
    ds.Configuration["Filtering"]["Enable Filters"] = False
    sc = ds.GetKDE_Scatter(yax="Defo", xax="Area")
    sc2 = ds.GetKDE_Scatter(yax="Defo", xax="Area",
                            positions=(ds.area_um, ds.deform))
    assert np.all(sc==sc2)
    

def test_empty_kde():
    ddict = example_data_dict(size=67, keys=["Area", "Defo"])
    ds = dclab.RTDC_DataSet(ddict=ddict)
    ds._filter[:] = 0
    a = ds.GetKDE_Scatter()
    assert len(a) == 0


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
