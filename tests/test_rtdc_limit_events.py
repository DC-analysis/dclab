#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

import dclab

from helper_methods import example_data_dict


def test_limit_simple():
    keys = ["area_um", "deform", "time", "frame"]
    ddict = example_data_dict(size=9999, keys=keys)
    ds = dclab.new_dataset(ddict)

    assert np.sum(ds.filter.all) == 9999
    filtflt = {"limit events": 800}

    cfg = {"filtering": filtflt}
    ds.config.update(cfg)
    ds.apply_filter()

    assert np.sum(ds.filter.all) == 800


def test_limit_equal():
    keys = ["area_um", "deform", "time", "frame"]
    ddict = example_data_dict(size=9999, keys=keys)
    ds = dclab.new_dataset(ddict)

    assert np.sum(ds.filter.all) == 9999
    filtflt = {"limit events": 9999}

    cfg = {"filtering": filtflt}
    ds.config.update(cfg)
    ds.apply_filter()

    assert np.sum(ds.filter.all) == 9999


def test_limit_above():
    keys = ["area_um", "deform", "time", "frame"]
    ddict = example_data_dict(size=9999, keys=keys)
    ds = dclab.new_dataset(ddict)

    assert np.sum(ds.filter.all) == 9999
    filtflt = {"limit events": 10000}

    cfg = {"filtering": filtflt}
    ds.config.update(cfg)
    ds.apply_filter()

    assert np.sum(ds.filter.all) == 9999


def test_downsample_nan():
    """Deal with nans"""
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=8472, keys=keys)
    ddict["area_um"][:4700] = np.nan
    ds = dclab.new_dataset(ddict)
    ds.config["filtering"]["limit events"] = 2000

    assert np.sum(ds.filter.all) == 8472

    ds.apply_filter()

    assert np.sum(ds.filter.all) == 2000
    # this is the general test; nans are included
    assert np.sum(np.isnan(ds["area_um"][ds.filter.all])) != 0
    # this is the exact test (e.g. for the random state used)
    assert np.sum(np.isnan(ds["area_um"][ds.filter.all])) == 1150


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
