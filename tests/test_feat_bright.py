#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from dclab import new_dataset
from dclab.features.bright import get_bright

from helper_methods import retrieve_data, cleanup


def test_simple_bright():
    ds = new_dataset(retrieve_data("rtdc_data_traces_video_bright.zip"))
    for ii in range(2, 7):
        # This stripped data set has only 7 video frames / contours
        img = ds["image"][ii]
        cont = ds["contour"][ii]
        avg, std = get_bright(cont=cont, img=img, ret_data="avg,sd")
        assert np.allclose(avg, ds["bright_avg"][ii])
        assert np.allclose(std, ds["bright_sd"][ii])
    cleanup()


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
