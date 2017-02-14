#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Test functions for loading traces
"""
from __future__ import print_function, unicode_literals

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


def test_trace_basic():
    ds = RTDC_DataSet(tdms_path = retreive_tdms(example_data_sets[1]))
    assert ds.trace.__repr__().count("<not loaded into memory>"), "traces should not be loaded into memory before first access"
    assert len(ds.trace) == 6
    assert np.allclose(np.average(ds.trace["FL1med"][0]), 287.08999999999997)


def test_trace_methods():
    ds = RTDC_DataSet(tdms_path = retreive_tdms(example_data_sets[1]))
    for k in list(ds.trace.keys()):
        assert  k in  [u'FL1med', u'FL2raw', u'FL2med', u'FL3med', u'FL1raw', u'FL3raw']
    for k in ds.trace:
        assert  k in  [u'FL1med', u'FL2raw', u'FL2med', u'FL3med', u'FL1raw', u'FL3raw']
    assert ds.trace.__repr__().count("<loaded into memory>")


def test_trace_not_available():
    ddict = example_data_dict(size=67, keys=["Area", "Defo"])
    ds = RTDC_DataSet(ddict=ddict)    
    assert ds.trace.__repr__() == "No trace data available!"
    assert len(ds.trace) == 0



if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
