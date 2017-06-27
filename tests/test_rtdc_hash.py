#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from dclab import new_dataset, dfn

from helper_methods import example_data_dict, retreive_tdms, example_data_sets



def test_hash_dict():
    ddict = example_data_dict()
    ds = new_dataset(ddict)
    assert hash(ds) == 5144097797485431247


def test_hash_hierarchy():
    tdms_path = retreive_tdms(example_data_sets[1])
    ds1 = new_dataset(tdms_path)
    ds2 = new_dataset(ds1)
    assert hash(ds2) == 7983493369597685688


def test_hash_tdms():
    tdms_path = retreive_tdms(example_data_sets[1])
    ds = new_dataset(tdms_path)
    assert hash(ds) == -5287680266361205240



if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
