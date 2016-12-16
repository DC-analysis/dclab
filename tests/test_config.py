#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import copy
import sys
from os.path import abspath, dirname, join

import numpy as np

# Add parent directory to beginning of path variable
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from dclab import config, RTDC_DataSet
import tempfile

from helper_methods import retreive_tdms, example_data_sets


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
    

def test_config_save_load():
    ## Download and extract data
    tdms_path = retreive_tdms(example_data_sets[0])
    ds = RTDC_DataSet(tdms_path)
    _fd, cfg_file = tempfile.mkstemp()
    config.save_config_file(cfg_file, ds.Configuration)
    loaded = config.load_config_file(cfg_file, capitalize=False)
    assert equals(loaded, ds.Configuration)


def test_config_dtype():
    a = config.get_config_entry_dtype("Filtering", "Enable Filters")
    assert a == bool

    a = config.get_config_entry_dtype("General", "Channel Width")
    assert a == float

    a = config.get_config_entry_dtype("General", "Unknown Variable")
    assert a == float


def test_config_choices():
    c1 = config.get_config_entry_choices("Plotting", "KDE")
    for c2 in ["None", "Gauss", "Multivariate"]:
        assert c2 in c1
    
    c3 = config.get_config_entry_choices("Plotting", "Axis X")
    assert len(c3) != 0
    assert "Defo" in c3

    c3 = config.get_config_entry_choices("Plotting", "Axis Y", ignore_axes=["Defo"])
    assert len(c3) != 0
    assert not "Defo" in c3
    
    c4 = config.get_config_entry_choices("Plotting", "Rows")
    assert "1" in c4
    
    c5 = config.get_config_entry_choices("Plotting", "Scatter Marker Size")
    assert "1" in c5
    
    c6 = config.get_config_entry_choices("Plotting", "Scale Axis")
    assert "Linear" in c6    
    
    
def test_backwards_compatible_channel_width():
    cfg = copy.deepcopy(config.cfg)
    fd, fname = tempfile.mkstemp()
    cfg["General"].pop("Channel Width")
    cfg["General"]["Flow Rate [ul/s]"] = 0.16
    config.save_config_file(fname, cfg)
    cfg2 = config.load_config_file(fname)
    assert cfg2["General"]["Channel Width"] == 30
    

def test_backwards_compatible_circularity():
    cfg = copy.deepcopy(config.cfg)
    a = .01
    cfg["Plotting"]["Contour Accuracy Defo"] = a
    plotd = {"Contour Accuracy Circ":a*2}
    newcfg = {"Plotting":plotd}
    
    config.update_config_dict(cfg, newcfg)
    assert cfg["Plotting"]["Contour Accuracy Defo"] == a*2
    assert cfg["Plotting"]["Contour Accuracy Defo"] != a

if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
