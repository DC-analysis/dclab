#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import codecs
import numpy as np
import os
from os.path import abspath, dirname, join
import shutil
import sys
import warnings
import zipfile


# Add parent directory to beginning of path variable
sys.path.insert(0, dirname(dirname(abspath(__file__))))
import dclab

from helper_methods import retreive_tdms, example_data_sets



def test_export():    
    ## Download and extract data
    tdmsfile = retreive_tdms(example_data_sets[0])

    ds = dclab.RTDC_DataSet(tdmsfile)
    
    edest = dirname(dirname(tdmsfile))
    
    f1 = join(edest, "test.tsv")
    f2 = join(edest, "test_unicode.tsv")
    
    ds.ExportTSV(f1, ["Area", "Defo", "Time", "Frame", "FL-3width"], override=True)
    ds.ExportTSV(f2, [u"Area", u"Defo", u"Time", u"Frame", u"FL-3width"], override=True)
    
    with codecs.open(f1) as fd:
        a1 = fd.read()
    
    with codecs.open(f2) as fd:
        a2 = fd.read()

    assert a1 == a2
    assert len(a1) != 0

    # cleanup
    shutil.rmtree(edest, ignore_errors=True)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
