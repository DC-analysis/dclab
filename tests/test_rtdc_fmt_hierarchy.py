#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Test filter hierarchies
"""
from __future__ import print_function

import numpy as np

from dclab import new_dataset

from helper_methods import example_data_dict, example_data_sets, cleanup, \
    retrieve_data


def test_event_count():
    tdms_path = retrieve_data(example_data_sets[1])
    ds = new_dataset(tdms_path)
    ds.filter.manual[0] = False
    ch = new_dataset(ds)
    assert ds.config["experiment"]["event count"] == len(ds)
    assert ch.config["experiment"]["event count"] == len(ch)
    assert len(ds) == len(ch) + 1


def test_feat_contour():
    path = retrieve_data("rtdc_data_hdf5_contour_image_trace.zip")
    ds = new_dataset(path)
    ds.filter.manual[0] = False
    ds.filter.manual[2] = False
    ch = new_dataset(ds)
    assert np.all(ch["contour"][0] == ds["contour"][1])
    assert np.all(ch["contour"][1] == ds["contour"][3])


def test_feat_image():
    path = retrieve_data("rtdc_data_hdf5_contour_image_trace.zip")
    ds = new_dataset(path)
    ds.filter.manual[0] = False
    ds.filter.manual[2] = False
    ch = new_dataset(ds)
    assert np.all(ch["image"][0] == ds["image"][1])
    assert np.all(ch["image"][1] == ds["image"][3])


def test_feat_trace():
    path = retrieve_data("rtdc_data_hdf5_contour_image_trace.zip")
    ds = new_dataset(path)
    ds.filter.manual[0] = False
    ds.filter.manual[2] = False
    ch = new_dataset(ds)
    assert np.all(ch["trace"]["fl1_median"][0]
                  == ds["trace"]["fl1_median"][1])
    assert np.all(ch["trace"]["fl1_median"][1]
                  == ds["trace"]["fl1_median"][3])


def test_hierarchy_from_tdms():
    tdms_path = retrieve_data(example_data_sets[1])
    ds1 = new_dataset(tdms_path)
    ds2 = new_dataset(ds1)

    ds1.filter.manual[0] = False
    ds2.apply_filter()
    assert ds2._filter.shape[0] == ds1._filter.shape[0] - 1
    assert ds2["area_um"][0] == ds1["area_um"][1]
    cleanup()


def test_manual_exclude():
    data = example_data_dict(42, keys=["area_um", "deform"])
    p = new_dataset(data)
    c1 = new_dataset(p)
    c2 = new_dataset(c1)
    c3 = new_dataset(c2)
    c1.filter.manual[0] = False
    c2.apply_filter()
    c2.filter.manual[1] = False
    c3.apply_filter()
    c3.filter.manual[0] = False

    # simple exclusion of few events
    assert len(c3) == len(p) - 2

    # removing event in parent removes the event from the
    # child altogether, including the manual filter
    c2.filter.manual[0] = False
    c3.apply_filter()
    assert np.alltrue(c3.filter.manual)

    # reinserting the event in the parent, retrieves back
    # the manual filter in the child
    c2.filter.manual[0] = True
    c3.apply_filter()
    assert not c3.filter.manual[0]
    assert not c3.filter.manual[0]


def test_same_hash_different_identifier():
    tdms_path = retrieve_data(example_data_sets[1])
    ds1 = new_dataset(tdms_path)
    ds1.filter.manual[0] = False
    ch1 = new_dataset(ds1)
    ch2 = new_dataset(ds1)
    assert len(ch1) == len(ds1) - 1
    assert ch1.hash == ch2.hash
    assert ch1.identifier != ch2.identifier


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
