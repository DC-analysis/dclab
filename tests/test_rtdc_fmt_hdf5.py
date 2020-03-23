#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test hdf5 file format"""
from __future__ import print_function

import os

import h5py
import numpy as np
import pytest

from dclab import new_dataset, rtdc_dataset

from helper_methods import retrieve_data, cleanup


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_config():
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_contour_image_trace.zip"))
    assert ds.config["setup"]["channel width"] == 30
    assert ds.config["setup"]["chip region"].lower() == "channel"
    assert ds.config["setup"]["flow rate"] == 0.16
    assert ds.config["imaging"]["pixel size"] == 0.34
    cleanup()


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_contour_basic():
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_contour_image_trace.zip"))
    assert len(ds) == 5
    assert len(ds["contour"]) == 5
    assert np.allclose(np.average(ds["contour"][0]), 30.75)
    cleanup()


def test_defective_feature_aspect():
    # see https://github.com/ZELLMECHANIK-DRESDEN/ShapeOut/issues/241
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    # modify aspect feature
    with h5py.File(h5path, "a") as h5:
        aspect0 = h5["events/aspect"][0]
        aspect1 = 1.234
        assert not np.allclose(aspect0, aspect1), "test's sanity"
        h5["events/aspect"][0] = aspect1
        # In Shape-In 2.0.5 everything worked fine
        h5.attrs["setup:software version"] = "ShapeIn 2.0.5"
    # sanity check
    with new_dataset(h5path) as ds1:
        assert np.allclose(ds1["aspect"][0], aspect1)
    # trigger recomputation of aspect feature
    with h5py.File(h5path, "a") as h5:
        h5.attrs["setup:software version"] = "ShapeIn 2.0.6"
    # verify original value of aspect
    with new_dataset(h5path) as ds2:
        assert np.allclose(ds2["aspect"][0], aspect0)
    cleanup()


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_hash():
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_contour_image_trace.zip"))
    assert ds.hash == "2c436daba22d2c7397b74d53d80f8931"
    assert ds.format == "hdf5"
    cleanup()


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_image_basic():
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_contour_image_trace.zip"))
    assert np.allclose(np.average(ds["image"][1]), 125.37133333333334)
    assert len(ds["image"]) == 5
    cleanup()


def test_logs():
    path_in = retrieve_data("rtdc_data_hdf5_mask_contour.zip")

    with new_dataset(path_in) as ds:
        assert not ds.logs

    # write some logs
    with h5py.File(path_in, "a") as h5:
        rtdc_dataset.write(h5,
                           logs={"test_log": ["peter", "hans"]},
                           mode="append")

    with new_dataset(path_in) as ds:
        assert ds.logs
        assert ds.logs["test_log"][0] == "peter"

    # remove logs
    with h5py.File(path_in, "a") as h5:
        del h5["logs"]

    with new_dataset(path_in) as ds:
        assert not ds.logs
        try:
            ds.logs["test_log"]
        except KeyError:  # no log data
            pass
    cleanup()


def test_no_suffix():
    """Loading an .rtdc file that has a wrong suffix"""
    path = str(retrieve_data("rtdc_data_hdf5_mask_contour.zip"))
    path2 = path + ".wrong_suffix"
    os.rename(path, path2)
    ds = new_dataset(path2)
    assert(len(ds) == 8)


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
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
