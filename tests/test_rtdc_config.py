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

from helper_methods import example_data_dict, retreive_tdms, example_data_sets, cleanup


def equals(a, b):
    """Compare objects with allclose"""
    if isinstance(a, (dict, Configuration, CaseInsensitiveDict)):
        for key in a:
            assert key in b, "key not in b"
            assert equals(a[key], b[key])
    elif isinstance(a, (float, int)):
        assert np.allclose(a,b)
    else:
        assert a==b
    return True


def test_config_basic():
    ds = new_dataset(retreive_tdms(example_data_sets[1]))
    assert ds.config["roi"]["height"] == 96.
    cleanup()


def test_config_save_load():
    ## Download and extract data
    tdms_path = retreive_tdms(example_data_sets[0])
    ds = new_dataset(tdms_path)
    _fd, cfg_file = tempfile.mkstemp()
    ds.config.save(cfg_file)
    loaded = Configuration(files=[cfg_file])
    assert equals(loaded, ds.config)
    cleanup()
    
    
def test_backwards_compatible_channel_width():
    cfg = Configuration()
    fd, fname = tempfile.mkstemp()
    cfg["General"].pop("Channel Width")
    cfg["General"]["Flow Rate [ul/s]"] = 0.16
    cfg.save(fname)
    cfg2 = Configuration(files=[fname])
    assert cfg2["General"]["Channel Width"] == 30
    

if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
