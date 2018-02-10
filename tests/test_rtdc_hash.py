#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from dclab import new_dataset

from helper_methods import example_data_dict, retrieve_data, \
    example_data_sets, cleanup


def test_hash_dict():
    ddict = example_data_dict()
    ds = new_dataset(ddict)
    assert ds.hash == "aa4d0faf17bccadc474ad0c1fea4346e"


def test_hash_hierarchy():
    tdms_path = retrieve_data(example_data_sets[1])
    ds1 = new_dataset(tdms_path)
    ds2 = new_dataset(ds1)
    assert ds2.hash == "3e942ba6e1cb333d3607edaba5f2c618"
    cleanup()


def test_hash_tdms():
    tdms_path = retrieve_data(example_data_sets[1])
    ds = new_dataset(tdms_path)
    assert ds.hash == "92601489292dc9bf9fc040f87d9169c0"
    cleanup()


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
