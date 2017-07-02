#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import io
import numpy as np
import os
from os.path import abspath, dirname, join
import pytest
import shutil
import sys
import tempfile
import warnings
import zipfile


# Add parent directory to beginning of path variable
sys.path.insert(0, dirname(dirname(abspath(__file__))))
import dclab
from dclab import new_dataset


from helper_methods import example_data_dict, retreive_tdms, example_data_sets, cleanup


def test_avi_export():
    ds = new_dataset(retreive_tdms(example_data_sets[1]))
    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.avi")
    ds.export.avi(path=f1)
    assert os.stat(f1)[6] > 1e4, "Resulting file to small, Something went wrong!"
    shutil.rmtree(edest, ignore_errors=True)
    cleanup()


def test_avi_override():
    ds = new_dataset(retreive_tdms(example_data_sets[1]))
    
    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.avi")
    ds.export.avi(f1, override=True)
    try:
        ds.export.avi(f1[:-4], override=False)
    except OSError:
        pass
    else:
        raise ValueError("Should append .avi and not override!")

    # cleanup
    shutil.rmtree(edest, ignore_errors=True)
    cleanup()


def test_avi_no_images():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=127, keys=keys)
    ds = dclab.new_dataset(ddict)
    
    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.avi")
    try:
        ds.export.avi(f1)
    except OSError:
        pass
    else:
        raise ValueError("There should be no image data to write!")
    shutil.rmtree(edest, ignore_errors=True)


def test_fcs_export():    
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=222, keys=keys)
    ds = dclab.new_dataset(ddict)
    
    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.fcs")
    f2 = join(edest, "test_unicode.fcs")
    
    ds.export.fcs(f1, keys, override=True)
    ds.export.fcs(f2, [u"area_um", u"deform", u"time", u"frame", u"fl3_width"], override=True)
    
    with io.open(f1, mode="rb") as fd:
        a1 = fd.read()
    
    with io.open(f2, mode="rb") as fd:
        a2 = fd.read()

    assert a1 == a2
    assert len(a1) != 0

    # cleanup
    shutil.rmtree(edest, ignore_errors=True)


def test_fcs_override():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=212, keys=keys)
    ds = dclab.new_dataset(ddict)
    
    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.fcs")
    ds.export.fcs(f1, keys, override=True)
    try:
        ds.export.fcs(f1[:-4], keys, override=False)
    except OSError:
        pass
    else:
        raise ValueError("Should append .fcs and not override!")

    # cleanup
    shutil.rmtree(edest, ignore_errors=True)


def test_fcs_not_filtered():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=127, keys=keys)
    ds = dclab.new_dataset(ddict)
    
    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.tsv")
    ds.export.fcs(f1, keys, filtered=False)

    # cleanup
    shutil.rmtree(edest, ignore_errors=True)


def test_tsv_export():    
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=222, keys=keys)
    ds = dclab.new_dataset(ddict)
    
    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.tsv")
    f2 = join(edest, "test_unicode.tsv")
    
    ds.export.tsv(f1, keys, override=True)
    ds.export.tsv(f2, [u"area_um", u"deform", u"time", u"frame", u"fl3_width"], override=True)
    
    with io.open(f1) as fd:
        a1 = fd.read()
    
    with io.open(f2) as fd:
        a2 = fd.read()

    assert a1 == a2
    assert len(a1) != 0

    # cleanup
    shutil.rmtree(edest, ignore_errors=True)


def test_tsv_override():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=212, keys=keys)
    ds = dclab.new_dataset(ddict)
    
    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.tsv")
    ds.export.tsv(f1, keys, override=True)
    try:
        ds.export.tsv(f1[:-4], keys, override=False)
    except OSError:
        pass
    else:
        raise ValueError("Should append .tsv and not override!")

    # cleanup
    shutil.rmtree(edest, ignore_errors=True)


def test_tsv_not_filtered():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=127, keys=keys)
    ds = dclab.new_dataset(ddict)
    
    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.tsv")
    ds.export.tsv(f1, keys, filtered=False)

    # cleanup
    shutil.rmtree(edest, ignore_errors=True)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
