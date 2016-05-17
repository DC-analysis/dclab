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

from helper_methods import retreive_tdms, example_data_sets

def test_kde_general():
    ## Download and extract data
    tdmsfile = retreive_tdms(example_data_sets[0])

    ds = dclab.RTDC_DataSet(tdmsfile)
    
    kdes = dclab.dfn.GetParameterChoices("Plotting", "KDE")
    
    for kde in kdes:
        ds.Configuration["Plotting"]["KDE"] = kde
        ds.GetKDE_Contour()
        ds.GetKDE_Scatter()


def test_kde_none():
    tdmsfile = retreive_tdms(example_data_sets[0])

    ds = dclab.RTDC_DataSet(tdmsfile)
    ds.Configuration["Plotting"]["KDE"] = "None"
    sc = ds.GetKDE_Scatter()
    assert np.sum(sc) == sc.shape[0]



if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
