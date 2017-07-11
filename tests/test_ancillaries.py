#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function


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

from helper_methods import example_data_dict, retreive_tdms, example_data_sets, cleanup


def test_0basic():
    ds = dclab.new_dataset(retreive_tdms(example_data_sets[1]))
    for cc in [  
                'fl1_pos',
                'frame',
                'size_x',
                'size_y',
                'contour',
                'area_cvx',
                'circ',
                'image',
                'trace',
                'fl1_width',
                'ncells',
                'pos_x',
                'pos_y',
                'fl1_area',
                'fl1_max',
                ]:
        assert cc in ds

    # ancillaries
    for cc in [ 
               "deform",
               "area_um",
               "aspect",
               "frame",
               "index",
               "time",
               ]:
        assert cc in ds
    
    cleanup()


def test_0error():
    keys = ["circ"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    try:
        ds["unknown_column"]
    except KeyError:
        pass
    else:
        raise ValueError("Should have raised KeyError!")


def test_aspect():
    # Aspect ratio of the data
    ds = dclab.new_dataset(retreive_tdms("rtdc_data_traces_video_bright.zip"))
    aspect = ds["aspect"]
    assert np.sum(aspect>1) == 904
    assert np.sum(aspect<1) == 48
    cleanup() 


def test_area_ratio():
    ds = dclab.new_dataset(retreive_tdms("rtdc_data_traces_video.zip"))
    comp_ratio = ds["area_ratio"]
    # The convex area is always >= the raw area
    assert np.all(comp_ratio>=1)
    assert np.allclose(comp_ratio[0], 1.0196464)
    cleanup() 


def test_brightness():
    # Brightness of the image
    ds = dclab.new_dataset(retreive_tdms("rtdc_data_traces_video_bright.zip"))
    # This is something low-level and should not be done in a script.
    # Remove the brightness columns from RTDCBase to force computation with
    # the image and contour columns. 
    real_avg = ds._events.pop("bright_avg")
    real_sd = ds._events.pop("bright_sd")
    # This will cause a zero-padding warning:
    comp_avg = ds["bright_avg"]
    comp_sd = ds["bright_sd"] 
    idcompare = ~np.isnan(comp_avg)
    # ignore first event (no image data)
    idcompare[0] = False
    assert np.allclose(real_avg[idcompare], comp_avg[idcompare])
    assert np.allclose(real_sd[idcompare], comp_sd[idcompare])
    cleanup()


def test_deform():
    keys = ["circ"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    assert np.allclose(ds["deform"], 1-ds["circ"])
    

def test_emodulus():
    keys = ["area_um", "deform"]
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


def test_emodulus_area():
    # computes "area_um" from "area_cvx"
    keys = ["area_cvx", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    # area can be computed from areapix
    assert "area_um" in ds
    assert "emodulus" not in ds, "not config for emodulus"
    ds.config["calculation"] = {"emodulus medium": "CellCarrier",
                                "emodulus model": "elastic sphere",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    assert "emodulus" in ds


def test_emodulus_none():
    keys = ["area_msd", "deform"]
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
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    assert "emodulus" not in ds, "not config for emodulus"
    ds.config["calculation"] = {"emodulus medium": "CellCarrier",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    assert "emodulus" not in ds, "emodulus model should be missing"


def test_time():
    ds = dclab.new_dataset(retreive_tdms("rtdc_data_minimal.zip"))
    tt = ds["time"]
    assert tt[0] == 0
    assert np.allclose(tt[1], 0.0385)
    assert np.all(np.diff(tt) > 0)
    cleanup()


def test_volume():
    ds = dclab.new_dataset(retreive_tdms("rtdc_data_minimal.zip"))
    vol = ds["volume"]
    # There are a lot of nans, because the contour is not given everywhere
    vol = vol[~np.isnan(vol)]
    assert np.allclose(vol[0], 574.60368907528346)
    assert np.allclose(vol[12], 1010.5669523203878)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()