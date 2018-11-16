#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Downsampling only affects RTDCBase._plot_filter"""
from __future__ import print_function

import numpy as np

import dclab

from helper_methods import example_data_dict


def test_downsample_log():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)

    ds.apply_filter()
    xlin, ylin = ds.get_downsampled_scatter(downsample=100,
                                            xscale="linear",
                                            yscale="linear")
    xlog, ylog = ds.get_downsampled_scatter(downsample=100,
                                            xscale="log",
                                            yscale="log")
    assert not np.all(xlin == xlog)
    assert not np.all(ylin == ylog)
    for x in xlog:
        assert x in ds["area_um"]
    for y in ylog:
        assert y in ds["deform"]


def test_downsample_log_invalid():
    ddict = {"area_um": np.linspace(-2, 10, 100),
             "deform": np.linspace(.1, .5, 100)}
    ds = dclab.new_dataset(ddict)
    ds.apply_filter()
    xlog, ylog = ds.get_downsampled_scatter(downsample=99,
                                            xscale="log",
                                            yscale="log")
    for x in xlog:
        assert x in ds["area_um"]
    for y in ylog:
        assert y in ds["deform"]

    assert xlog.size == 99
    assert ylog.size == 99
    assert xlog.min() < 0


def test_downsample_none():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)

    assert np.sum(ds._plot_filter) == 8472
    ds.apply_filter()
    ds.get_downsampled_scatter(downsample=0)
    assert np.sum(ds._plot_filter) == 8472


def test_downsample_none2():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)

    assert np.sum(ds._plot_filter) == 8472

    filtflt = {"enable filters": False}

    cfg = {"filtering": filtflt}
    ds.config.update(cfg)
    ds.apply_filter()
    ds.get_downsampled_scatter(downsample=100)

    assert np.sum(ds._plot_filter) == 100
    assert np.sum(ds._filter) == 8472

    filtflt["enable filters"] = True
    ds.config.update(cfg)
    ds.apply_filter()
    ds.get_downsampled_scatter(downsample=100)

    assert np.sum(ds._plot_filter) == 100
    assert np.sum(ds._filter) == 8472


def test_downsample_yes():
    """Simple downsampling test"""
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)

    assert np.sum(ds._plot_filter) == 8472

    ds.apply_filter()
    ds.get_downsampled_scatter(downsample=100)
    assert np.sum(ds._plot_filter) == 100
    ds.get_downsampled_scatter(downsample=100)


def test_downsample_up():
    """
    Likely causes removal of too many points and requires
    re-inserting them.
    """
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=10000, keys=keys)
    ds = dclab.new_dataset(ddict)

    assert np.sum(ds._plot_filter) == 10000

    ds.apply_filter()
    ds.get_downsampled_scatter(downsample=9999)
    assert np.sum(ds._plot_filter) == 9999
    ds.get_downsampled_scatter(downsample=9999)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
