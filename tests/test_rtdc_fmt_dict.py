#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Test functions for loading traces
"""
from __future__ import print_function, unicode_literals

from dclab import new_dataset

from helper_methods import example_data_dict


def test_contour_not_available():
    ddict = example_data_dict(size=67, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    assert "contour" not in ds


def test_image_not_available():
    ddict = example_data_dict(size=67, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    assert "image" not in ds


def test_min_max_update():
    ddict = example_data_dict(size=67, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    cfg = ds.config.copy()

    # Force updating circularity
    cfg["filtering"]["defo min"] = .4
    cfg["filtering"]["defo max"] = .8
    ds.config.update(cfg)

    ds.apply_filter()


def test_trace_not_available():
    ddict = example_data_dict(size=67, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    assert "trace" not in ds


def test_wrong_things():
    ddict = example_data_dict(size=67, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    # Check unknown variable (warning will be displayed
    try:
        ds.apply_filter(force=["on_purpose_unknown"])
    except ValueError:
        pass


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
