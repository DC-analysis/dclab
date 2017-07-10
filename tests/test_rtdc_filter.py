#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Test functions for loading contours
"""
from __future__ import print_function, unicode_literals

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
from dclab.rtdc_dataset import new_dataset
from dclab.rtdc_dataset.config import Configuration, CaseInsensitiveDict

from helper_methods import example_data_dict



def test_filter_min_max():
    # make sure min/max values are filtered
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    amin, amax = ds["area_um"].min(), ds["area_um"].max()
    ds.config["filtering"]["area_um min"] = (amax+amin)/2
    ds.config["filtering"]["area_um max"] = amax
    ds.apply_filter()
    assert np.sum(ds.filter.all) == 4256
    
    ds.config["filtering"]["deform min"] = ds["deform"].min()
    ds.config["filtering"]["deform max"] = ds["deform"].max()
    assert np.sum(ds.filter.all) == 4256


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
