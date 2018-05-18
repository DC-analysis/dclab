#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test hdf5 file format"""
from __future__ import print_function

import numpy as np
import pytest

from dclab import new_dataset

from helper_methods import retrieve_data, cleanup


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.fmt_hdf5.UnknownKeyWarning')
def test_config():
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_contour_image_trace.zip"))
    assert ds.config["setup"]["channel width"] == 30
    assert ds.config["setup"]["chip region"].lower() == "channel"
    assert ds.config["setup"]["flow rate"] == 0.16
    assert ds.config["imaging"]["pixel size"] == 0.34
    cleanup()


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.fmt_hdf5.UnknownKeyWarning')
def test_contour_basic():
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_contour_image_trace.zip"))
    assert len(ds) == 5
    assert len(ds["contour"]) == 5
    assert np.allclose(np.average(ds["contour"][0]), 30.75)
    cleanup()


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.fmt_hdf5.UnknownKeyWarning')
def test_hash():
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_contour_image_trace.zip"))
    assert ds.hash == "2c436daba22d2c7397b74d53d80f8931"
    assert ds.format == "hdf5"
    cleanup()


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.fmt_hdf5.UnknownKeyWarning')
def test_image_basic():
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_contour_image_trace.zip"))
    assert np.allclose(np.average(ds["image"][1]), 125.37133333333334)
    assert len(ds["image"]) == 5
    cleanup()


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.fmt_hdf5.UnknownKeyWarning')
def test_trace():
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_contour_image_trace.zip"))
    assert len(ds["trace"]) == 2
    assert ds["trace"]["fl1_raw"].shape == (5, 100)
    assert np.allclose(np.average(
        ds["trace"]["fl1_median"][0]), 0.027744706519425219)
    cleanup()


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in sorted(list(loc.keys())):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
