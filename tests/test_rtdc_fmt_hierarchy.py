#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Test filter hierarchies
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
from dclab import new_dataset, dfn

from helper_methods import example_data_dict, retreive_tdms, example_data_sets, cleanup


def test_hierarchy_from_tdms():
    tdms_path = retreive_tdms(example_data_sets[1])
    ds1 = new_dataset(tdms_path)
    ds2 = new_dataset(ds1)

    ds1.filter.manual[0] = False
    ds2.apply_filter()
    assert ds2._filter.shape[0] == ds1._filter.shape[0]-1
    assert ds2["area_um"][0] == ds1["area_um"][1]
    cleanup()


def test_same_hash_different_identifier():
    tdms_path = retreive_tdms(example_data_sets[1])
    ds1 = new_dataset(tdms_path)
    ds1.filter.manual[0] = False
    ch1 = new_dataset(ds1)
    ch2 = new_dataset(ds1)
    assert len(ch1) == len(ds1)-1
    assert ch1.hash == ch2.hash
    assert ch1.identifier != ch2.identifier


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
