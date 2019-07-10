#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals

import numpy as np

from dclab.rtdc_dataset import new_dataset

from helper_methods import example_data_dict


def test_filter_manual():
    # make sure min/max values are filtered
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    ds.filter.manual[[0, 8471]] = False
    ds.apply_filter()
    assert len(ds["deform"][ds.filter.all]) == 8470
    assert ds["deform"][1] == ds["deform"][ds.filter.all][0]


def test_filter_min_max():
    # make sure min/max values are filtered
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    amin, amax = ds["area_um"].min(), ds["area_um"].max()
    ds.config["filtering"]["area_um min"] = (amax + amin) / 2
    ds.config["filtering"]["area_um max"] = amax
    ds.apply_filter()
    assert np.sum(ds.filter.all) == 4256

    # make sure data is not filtered before calling ds.apply_filter
    dmin, dmax = ds["deform"].min(), ds["deform"].max()
    ds.config["filtering"]["deform min"] = (dmin + dmax) / 2
    ds.config["filtering"]["deform max"] = dmax
    assert np.sum(ds.filter.all) == 4256


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
