#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Test functions for loading contours
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


def equals(a, b):
    """Compare objects with allclose"""
    if isinstance(a, dict):
        for key in a:
            assert key in b, "key not in b"
            assert equals(a[key], b[key])
    elif isinstance(a, (float, int)):
        assert np.allclose(a,b)
    else:
        assert a==b
    return True



def test_config_basic():
    ds = RTDC_DataSet(tdms_path = retreive_tdms(example_data_sets[1]))


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
