#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
from os.path import abspath, dirname, join

import numpy as np

# Add parent directory to beginning of path variable
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from dclab import config, RTDC_DataSet
import tempfile

from helper_methods import retreive_tdms, example_data_sets


def test_config_save_load():
    ## Download and extract data
    tdms_path = retreive_tdms(example_data_sets[0])
    ds = RTDC_DataSet(tdms_path)
    _fd, cfg_file = tempfile.mkstemp()
    config.save_config_file(cfg_file, ds.Configuration)
    loaded = config.load_config_file(cfg_file, capitalize=False)
    assert loaded == ds.Configuration
    

if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
