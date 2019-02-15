#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals

import pytest

from dclab.rtdc_dataset import check_dataset

from helper_methods import cleanup, retrieve_data


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.fmt_hdf5.UnknownKeyWarning')
def test_basic():
    h5path = retrieve_data("rtdc_data_hdf5_contour_image_trace.zip")
    viol, aler, info = check_dataset(h5path)
    # Features: Unknown key 'ncells'
    # Metadata: Missing key [fluorescence] 'channels installed'
    # Metadata: Missing key [fluorescence] 'laser count'
    # Metadata: Missing key [fluorescence] 'lasers installed'
    # Metadata: Missing key [fluorescence] 'samples per event'
    # Metadata: Unknown key [imaging] 'exposure time'
    # Metadata: Unknown key [imaging] 'flash current'
    # Metadata: Unknown key [setup] 'temperature'
    # Metadata: fluorescence channel count inconsistent
    assert len(viol) == 9
    # Metadata: Missing key [setup] identifier'
    # Metadata: Missing section 'online_contour'
    # "HDF5: '/image': attribute 'CLASS' should be fixed-length ASCII string",
    # "HDF5: '/image': attribute 'IMAGE_SUBCLASS' should be fixed-length ...",
    # "HDF5: '/image': attribute 'IMAGE_VERSION' should be fixed-length ...",
    assert len(aler) == 5
    assert "Data file format: hdf5" in info
    assert "Fluorescence: True" in info
    cleanup()


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
