#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import codecs
import numpy as np
import os
from os.path import abspath, dirname, join
import shutil
import sys
import warnings
import zipfile


# Add parent directory to beginning of path variable
sys.path.insert(0, dirname(dirname(abspath(__file__))))
import dclab

from helper_methods import retreive_tdms, example_data_sets



def test_stat_defo():
    ## Download and extract data
    tdmsfile = retreive_tdms(example_data_sets[0])

    ds = dclab.RTDC_DataSet(tdmsfile)
    
    head, vals = dclab.statistics.get_statistics(ds, axes=["Defo"])
    
    for h, v in zip(head,vals):
        if h.lower() == "flow rate":
            assert v==0.12
        elif h.lower() == "events":
            assert v==5085
        elif h.lower() == "%-gated":
            assert v==100
        elif h.lower().startswith("sd "):
            assert np.allclose(v,0.04143419489264488)
        elif h.lower().startswith("median "):
            assert np.allclose(v,0.11600667238235474)
        elif h.lower().startswith("mode "):
            assert np.allclose(v,0.11187175661325455)
        elif h.lower().startswith("mean "):
            assert np.allclose(v,0.12089553475379944)
       
    # cleanup
    edest = dirname(dirname(tdmsfile))
    shutil.rmtree(edest, ignore_errors=True)


def test_stat_occur():
    ## Download and extract data
    tdmsfile = retreive_tdms(example_data_sets[0])

    ds = dclab.RTDC_DataSet(tdmsfile)
    
    head1, vals1 = dclab.statistics.get_statistics(ds, axes=["Defo"])
    head2, vals2 = dclab.statistics.get_statistics(ds, columns=["Events", "Mean"])
    headf, valsf = dclab.statistics.get_statistics(ds)
    
    # disable filtering (there are none anyway) to cover a couple more lines:
    ds.Configuration["Filtering"]["Enable Filters"] = False
    headn, valsn = dclab.statistics.get_statistics(ds)
    
    
    for item in zip(head1, vals1):
        assert item in zip(headf, valsf)

    for item in zip(head2, vals2):
        assert item in zip(headf, valsf)

    for item in zip(headn, valsn):
        assert item in zip(headf, valsf)

    # cleanup
    edest = dirname(dirname(tdmsfile))
    shutil.rmtree(edest, ignore_errors=True)



if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
