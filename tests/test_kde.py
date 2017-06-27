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
    ds = dclab.new_dataset(ddict)

    dcont = []    
    dscat = []
    for kde_type in dclab.kde_methods.methods:
        dcont.append(ds.get_kde_contour(kde_type=kde_type))
        dscat.append(ds.get_kde_scatter(kde_type=kde_type))
    
    for ii in range(1, len(dcont)-1):
        assert not np.allclose(dcont[ii], dcont[0])
        assert not np.allclose(dscat[ii], dscat[0])


def test_kde_none():
    ddict = example_data_dict()
    ds = dclab.new_dataset(ddict)
    sc = ds.get_kde_scatter(kde_type="none")
    assert np.sum(sc) == sc.shape[0]
    ds.get_kde_contour()


def test_kde_nofilt():
    ddict = example_data_dict()
    ds = dclab.new_dataset(ddict)
    ds.config["filtering"]["enable filters"] = False
    sc = ds.get_kde_scatter()
    cc = ds.get_kde_contour()
    assert sc.shape[0] == 100
    # This will fail if the default contour accuracy is changed
    # in `get_kde_contour`.
    assert cc[0].shape == (10,10)


def test_kde_positions():
    ddict = example_data_dict()
    ds = dclab.new_dataset(ddict)
    
    ds.config["filtering"]["enable filters"] = False
    sc = ds.get_kde_scatter(xax="area_um", yax="deform")
    sc2 = ds.get_kde_scatter(xax="area_um", yax="deform",
                             positions=(ds["area_um"], ds["deform"]))
    assert np.all(sc==sc2)


def test_empty_kde():
    ddict = example_data_dict(size=67, keys=["area_um", "deform"])
    ds = dclab.new_dataset(ddict)
    ds._filter[:] = 0
    a = ds.get_kde_scatter()
    assert len(a) == 0


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
