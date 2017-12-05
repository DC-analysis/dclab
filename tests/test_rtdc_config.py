#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Test functions for loading contours
"""
from __future__ import print_function, unicode_literals

import os
import tempfile

import numpy as np

from dclab.rtdc_dataset import new_dataset
from dclab.rtdc_dataset.config import Configuration, CaseInsensitiveDict

from helper_methods import retrieve_data, example_data_sets, cleanup


def equals(a, b):
    """Compare objects with allclose"""
    if isinstance(a, (dict, Configuration, CaseInsensitiveDict)):
        for key in a:
            assert key in b, "key not in b"
            assert equals(a[key], b[key])
    elif isinstance(a, (float, int)):
        if np.isnan(a):
            assert np.isnan(b)
        else:
            assert np.allclose(a, b), "a={} vs b={}".format(a, b)
    else:
        assert a == b, "a={} vs b={}".format(a, b)
    return True


def test_config_basic():
    ds = new_dataset(retrieve_data(example_data_sets[1]))
    assert ds.config["imaging"]["roi size y"] == 96.
    cleanup()


def test_config_save_load():
    # Download and extract data
    tdms_path = retrieve_data(example_data_sets[0])
    ds = new_dataset(tdms_path)
    cfg_file = tempfile.mktemp(prefix="test_dclab_rtdc_config_")
    ds.config.save(cfg_file)
    loaded = Configuration(files=[cfg_file])
    assert equals(loaded, ds.config)
    cleanup()
    try:
        os.remove(cfg_file)
    except OSError:
        pass


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
