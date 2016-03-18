#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
from os.path import abspath, dirname

import numpy as np

# Add parent directory to beginning of path variable
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import dclab

import os
from os.path import dirname, join, exists, isdir, abspath

import warnings
import zipfile

webloc = "https://github.com/ZellMechanik-Dresden/RTDCdata/raw/master/"

def dl_file(url, dest, chunk_size=6553):
    """
    Download `url` to `dest`.
    """
    import urllib3
    http = urllib3.PoolManager()
    r = http.request('GET', url, preload_content=False)
    with open(dest, 'wb') as out:
        while True:
            data = r.read(chunk_size)
            if data is None or len(data)==0:
                break
            out.write(data)
    r.release_conn()



def test_export():    
    ## Download and extract data
    file = "SimpleMeasurement.zip"
    url = join(webloc, file)
    dest = join(dirname(abspath(__file__)), file)
    # download
    dl_file(url, dest)
    # unpack
    arc = zipfile.ZipFile(dest)
    # extract all files to a directory with this filename
    edest = abspath(__file__)[:-3]
    arc.extractall(edest)
    
    ## Load RTDC Data set
    tdmsfile = join(edest, "Online/M1_2us_70A_0.120000ul_s.tdms")
    ds = dclab.RTDC_DataSet(tdmsfile)
    
    ds.ExportTSV(join(edest, "test"), ["Area", "Defo", "Time", "Frame", "FL-3width"], override=True)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
