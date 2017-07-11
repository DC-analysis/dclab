#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import os
from os.path import abspath, dirname, join
import shutil
import sys
import tempfile
import warnings
import zipfile

from helper_methods import retreive_tdms, cleanup

# Add parent directory to beginning of path variable
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from dclab import new_dataset
from dclab.brightness import get_brightness


def test_simple_bright():
    ds = new_dataset(retreive_tdms("rtdc_data_traces_video_bright.zip"))
    for ii in range(2,7):
        # This stripped data set has only 7 video frames / contours
        img = ds["image"][ii][:,:,0]
        cont = ds["contour"][ii]
        avg, std = get_brightness(cont=cont, img=img, ret_data="avg,sd")
        assert np.allclose(avg, ds["bright_avg"][ii])
        assert np.allclose(std, ds["bright_sd"][ii])
    cleanup()


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
