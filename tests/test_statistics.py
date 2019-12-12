#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import pytest

import dclab

from helper_methods import example_data_dict


def test_stat_simple():
    ddict = example_data_dict(size=5085, keys=["area_um", "tilt"])
    ds = dclab.new_dataset(ddict)

    head, vals = dclab.statistics.get_statistics(ds, features=["tilt"])

    for h, v in zip(head, vals):
        if h.lower() == "flow rate":
            assert np.isnan(v)  # backwards compatibility!
        elif h.lower() == "events":
            assert v == 5085
        elif h.lower() == "%-gated":
            assert v == 100
        elif h.lower().startswith("sd "):
            assert np.allclose(v, 0.288990352083)
        elif h.lower().startswith("median "):
            assert np.allclose(v, 0.494188566376)
        elif h.lower().startswith("mode "):
            assert np.allclose(v, 0.260923009639)
        elif h.lower().startswith("mean "):
            assert np.allclose(v, 0.497743857424)


def test_stat_occur():
    ddict = example_data_dict(size=5085, keys=["area_um", "deform"])
    ds = dclab.new_dataset(ddict)

    head1, vals1 = dclab.statistics.get_statistics(ds, features=["deform"])
    head2, vals2 = dclab.statistics.get_statistics(
        ds, methods=["Events", "Mean"])
    headf, valsf = dclab.statistics.get_statistics(ds)

    # disable filtering (there are none anyway) to cover a couple more lines:
    ds.config["filtering"]["enable filters"] = False
    headn, valsn = dclab.statistics.get_statistics(ds)

    for item in zip(head1, vals1):
        assert item in zip(headf, valsf)

    for item in zip(head2, vals2):
        assert item in zip(headf, valsf)

    for item in zip(headn, valsn):
        assert item in zip(headf, valsf)


def test_flow_rate():
    ddict = example_data_dict(size=77, keys=["area_um", "deform"])
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.172

    head1, vals1 = dclab.statistics.get_statistics(ds, features=["deform"])
    head2, vals2 = dclab.statistics.get_statistics(
        ds, methods=["Events", "Mean"])
    headf, valsf = dclab.statistics.get_statistics(ds)

    # disable filtering (there are none anyway) to cover a couple more lines:
    ds.config["filtering"]["enable filters"] = False
    headn, valsn = dclab.statistics.get_statistics(ds)

    for item in zip(head1, vals1):
        assert item in zip(headf, valsf)

    for item in zip(head2, vals2):
        assert item in zip(headf, valsf)

    for item in zip(headn, valsn):
        assert item in zip(headf, valsf)


@pytest.mark.filterwarnings('ignore::dclab.statistics.BadMethodWarning')
def test_false_method():
    def bad_method(x):
        return x + 1
    dclab.statistics.Statistics(name="bad",
                                req_feature=False,
                                method=bad_method)
    ddict = example_data_dict(size=77, keys=["area_um", "deform"])
    ds = dclab.new_dataset(ddict)
    head1, vals1 = dclab.statistics.get_statistics(ds, features=["deform"])
    out = {}
    for h, v in zip(head1, vals1):
        out[h] = v
    assert np.isnan(out["bad"])

    # clean up
    mth = dclab.statistics.Statistics.available_methods
    for k in mth:
        if k == "bad":
            mth.pop(k)
            break


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
