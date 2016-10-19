#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Test filter hierarchies
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
from dclab import RTDC_DataSet, dfn

from helper_methods import example_data_dict, retreive_tdms, example_data_sets


def test_hierarchy_from_tdms():
    tdms_path = retreive_tdms(example_data_sets[1])
    ds1 = RTDC_DataSet(tdms_path)
    ds2 = RTDC_DataSet(hparent=ds1)

    ds1._filter_manual[0] = False
    ds2.ApplyFilter()
    assert ds2._filter.shape[0] == ds1._filter.shape[0]-1
    assert ds2.area[0] == ds1.area[1]


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
