#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals

import warnings

import numpy as np

import dclab
from dclab.rtdc_dataset import new_dataset

from helper_methods import example_data_dict


def test_changed_polygon_filter():
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    amin, amax = ds["area_um"].min(), ds["area_um"].max()
    dmin, dmax = ds["deform"].min(), ds["deform"].max()
    pf = dclab.PolygonFilter(axes=["area_um", "deform"],
                             points=[[amin, dmin],
                                     [(amax + amin) / 2, dmin],
                                     [(amax + amin) / 2, dmax],
                                     ])
    ds.config["filtering"]["polygon filters"].append(pf.unique_id)
    ds.apply_filter()
    assert np.sum(ds.filter.all) == 2138
    # change the filter
    pf.points = list(pf.points) + [np.array([amin, dmax])]
    ds.apply_filter()
    assert np.sum(ds.filter.all) == 4215
    # invert the filter
    pf.inverted = True
    ds.apply_filter()
    assert np.sum(ds.filter.all) == 4257


def test_disable_filters():
    """Disabling the filters should only affect RTDCBase.filter.all"""
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    ds.filter.manual[[0, 8471]] = False
    ds.apply_filter()
    ds.config["filtering"]["enable filters"] = False
    ds.apply_filter()
    assert np.alltrue(ds.filter.all)


def test_filter_manual():
    # make sure min/max values are filtered
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    ds.filter.manual[[0, 8471]] = False
    ds.apply_filter()
    assert len(ds["deform"][ds.filter.all]) == 8470
    assert ds["deform"][1] == ds["deform"][ds.filter.all][0]


def test_filter_min_max():
    # make sure min/max values are filtered
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    amin, amax = ds["area_um"].min(), ds["area_um"].max()
    ds.config["filtering"]["area_um min"] = (amax + amin) / 2
    ds.config["filtering"]["area_um max"] = amax
    ds.apply_filter()
    assert np.sum(ds.filter.all) == 4256

    # make sure data is not filtered before calling ds.apply_filter
    dmin, dmax = ds["deform"].min(), ds["deform"].max()
    ds.config["filtering"]["deform min"] = (dmin + dmax) / 2
    ds.config["filtering"]["deform max"] = dmax
    assert np.sum(ds.filter.all) == 4256


def test_nan_warning():
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ddict["area_um"][[1, 4, 6]] = np.nan
    ds = new_dataset(ddict)
    amin, amax = np.nanmin(ds["area_um"]), np.nanmax(ds["area_um"])
    ds.config["filtering"]["area_um min"] = (amax + amin) / 2
    ds.config["filtering"]["area_um max"] = amax
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds.apply_filter()
        for wi in w:  # sometimes there are ResourceWarnings
            if wi.category == dclab.rtdc_dataset.filter.NanWarning:
                assert str(wi.message).count("area_um")
                break
        else:
            assert False, "Expected NanWarning"
    assert np.sum(ds.filter.all) == 4255


def test_only_one_boundary_error():
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ddict["area_um"][[1, 4, 6]] = np.nan
    ds = new_dataset(ddict)
    amin, amax = np.nanmin(ds["area_um"]), np.nanmax(ds["area_um"])
    ds.config["filtering"]["area_um min"] = (amax + amin) / 2
    try:
        ds.apply_filter()
    except ValueError:
        pass
    else:
        assert False, "setting only half of a box filter should not work"


def test_remove_ancillary_feature():
    """When a feature is removed, the box boolean filter must be deleted"""
    ddict = example_data_dict(size=8472, keys=["area_um", "deform",
                                               "emodulus"])
    ds = new_dataset(ddict)
    amin, amax = ds["area_um"].min(), ds["area_um"].max()
    ds.config["filtering"]["area_um min"] = (amax + amin) / 2
    ds.config["filtering"]["area_um max"] = amax
    emin, emax = ds["emodulus"].min(), ds["emodulus"].max()
    ds.config["filtering"]["emodulus min"] = (emax + emin) / 2
    ds.config["filtering"]["emodulus max"] = emax
    ds.apply_filter()
    numevents = np.sum(ds.filter.all)
    # now remove the ancillary feature
    ds._events.pop("emodulus")
    ds.apply_filter()
    numevents2 = np.sum(ds.filter.all)
    assert numevents != numevents2


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
