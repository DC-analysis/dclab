#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import codecs
import numpy as np
import os
from os.path import abspath, dirname, join
import shutil
import sys
import tempfile
import warnings
import zipfile


# Add parent directory to beginning of path variable
sys.path.insert(0, dirname(dirname(abspath(__file__))))
import dclab

from helper_methods import example_data_dict


def test_export():    
    keys = ["Area", "Defo", "Time", "Frame", "FL-3width"]
    ddict = example_data_dict(size=222, keys=keys)
    ds = dclab.RTDC_DataSet(ddict=ddict)
    
    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.tsv")
    f2 = join(edest, "test_unicode.tsv")
    
    ds.ExportTSV(f1, keys, override=True)
    ds.ExportTSV(f2, [u"Area", u"Defo", u"Time", u"Frame", u"FL-3width"], override=True)
    
    with codecs.open(f1) as fd:
        a1 = fd.read()
    
    with codecs.open(f2) as fd:
        a2 = fd.read()

    assert a1 == a2
    assert len(a1) != 0

    # cleanup
    shutil.rmtree(edest, ignore_errors=True)


def test_override():
    keys = ["Area", "Defo", "Time", "Frame", "FL-3width"]
    ddict = example_data_dict(size=212, keys=keys)
    ds = dclab.RTDC_DataSet(ddict=ddict)
    
    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.tsv")
    ds.ExportTSV(f1, keys, override=True)
    try:
        ds.ExportTSV(f1[:-4], keys, override=False)
    except:
        pass
    else:
        raise ValueError("Should append .tsv and not override!")

    # cleanup
    shutil.rmtree(edest, ignore_errors=True)


def test_not_filtered():
    keys = ["Area", "Defo", "Time", "Frame", "FL-3width"]
    ddict = example_data_dict(size=127, keys=keys)
    ds = dclab.RTDC_DataSet(ddict=ddict)
    
    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.tsv")
    ds.ExportTSV(f1, keys, filtered=False)

    # cleanup
    shutil.rmtree(edest, ignore_errors=True)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
