#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Test filter hierarchies
"""
from __future__ import print_function

from dclab import new_dataset

from helper_methods import retrieve_data, example_data_sets, cleanup


def test_event_count():
    tdms_path = retrieve_data(example_data_sets[1])
    ds = new_dataset(tdms_path)
    ds.filter.manual[0] = False
    ch = new_dataset(ds)
    assert ds.config["experiment"]["event count"] == len(ds)
    assert ch.config["experiment"]["event count"] == len(ch)
    assert len(ds) == len(ch) + 1


def test_hierarchy_from_tdms():
    tdms_path = retrieve_data(example_data_sets[1])
    ds1 = new_dataset(tdms_path)
    ds2 = new_dataset(ds1)

    ds1.filter.manual[0] = False
    ds2.apply_filter()
    assert ds2._filter.shape[0] == ds1._filter.shape[0] - 1
    assert ds2["area_um"][0] == ds1["area_um"][1]
    cleanup()


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
