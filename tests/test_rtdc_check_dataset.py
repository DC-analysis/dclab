#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals

from dclab.rtdc_dataset import check_dataset

from helper_methods import cleanup, retrieve_data


def test_basic():
    h5path = retrieve_data("rtdc_data_hdf5_contour_image_trace.zip")
    viol, aler, info = check_dataset(h5path)
    # "Features: Unknown key 'ncells'"
    # Metadata: Missing key [fluorescence] channels installed
    # Metadata: Missing key [fluorescence] laser count
    # Metadata: Missing key [fluorescence] lasers installed
    # Metadata: Missing key [fluorescence] samples per event
    # Metadata: fluorescence channel count inconsitent
    assert len(viol) == 6
    # Metadata: Missing key [setup] identifier'
    # Metadata: Missing section 'online_contour'
    assert len(aler) == 2
    assert "Data file format: hdf5" in info
    assert "Fluorescence: True" in info
    cleanup()


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
