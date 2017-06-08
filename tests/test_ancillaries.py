#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function


import codecs
import numpy as np
import os
from os.path import abspath, dirname, join
import shutil
import sys
import time
import warnings
import zipfile


# Add parent directory to beginning of path variable
sys.path.insert(0, dirname(dirname(abspath(__file__))))
import dclab

from helper_methods import example_data_dict


def test_emodulus():
    keys = ["area", "defo"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    ds.config["calculation"] = {"emodulus medium": "CellCarrier",
                                "emodulus model": "elastic sphere",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    t1 = time.time()
    assert "emodulus" in ds
    t2 = time.time()
    
    t3 = time.time()
    ds["emodulus"]
    t4 = time.time()
    assert t4-t3 > t2-t1


def test_area_emodulus():
    # computes "area" from "areapix"
    keys = ["areapix", "defo"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    # area can be computed from areapix
    assert "area" in ds
    assert "emodulus" not in ds, "not config for emodulus"
    ds.config["calculation"] = {"emodulus medium": "CellCarrier",
                                "emodulus model": "elastic sphere",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    assert "emodulus" in ds


def test_emodulus_none():
    keys = ["arearaw", "defo"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    assert "emodulus" not in ds, "not config for emodulus"
    ds.config["calculation"] = {"emodulus medium": "CellCarrier",
                                "emodulus model": "elastic sphere",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    assert "emodulus" not in ds, "column 'area_um' should be missing"


def test_emodulus_none2():
    keys = ["area", "defo"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    assert "emodulus" not in ds, "not config for emodulus"
    ds.config["calculation"] = {"emodulus medium": "CellCarrier",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    assert "emodulus" not in ds, "emodulus model should be missing"


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()