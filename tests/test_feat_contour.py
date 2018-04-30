#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from dclab import new_dataset
from dclab.features.contour import get_contour

from helper_methods import retrieve_data, cleanup


def test_simple_contour():
    ds = new_dataset(retrieve_data("rtdc_data_traces_video_bright.zip"))
    # Note: contour "3" in ds is bad!
    cin = ds["contour"][2]
    mask = np.zeros_like(ds["image"][2], dtype="bool")
    mask[cin[:, 1], cin[:, 0]] = True
    cout = get_contour(mask)
    # length
    assert len(cin) == len(cout)
    # simple presence test
    for ci in cin:
        assert ci in cout
    # order
    for ii in range(1, len(cin)):
        c2 = np.roll(cin, ii, axis=0)
        if np.all(c2 == cout):
            break
    else:
        assert False, "contours not matching, check orientation?"
    cleanup()


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
