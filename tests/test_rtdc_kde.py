#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import dclab

from helper_methods import example_data_dict


def test_kde_empty():
    ddict = example_data_dict(size=67, keys=["area_um", "deform"])
    ds = dclab.new_dataset(ddict)
    ds.filter.all[:] = 0
    a = ds.get_kde_scatter()
    assert len(a) == 0


def test_kde_general():
    # Download and extract data
    ddict = example_data_dict()
    ds = dclab.new_dataset(ddict)

    dcont = []
    dscat = []
    for kde_type in dclab.kde_methods.methods:
        dcont.append(ds.get_kde_contour(kde_type=kde_type))
        dscat.append(ds.get_kde_scatter(kde_type=kde_type))

    for ii in range(1, len(dcont) - 1):
        assert not np.allclose(dcont[ii], dcont[0])
        assert not np.allclose(dscat[ii], dscat[0])


def test_kde_linear_scatter():
    ddict = example_data_dict(size=300, keys=["area_um", "deform"])
    ddict["deform"][:20] = .1
    ddict["area_um"][:20] = .5
    ds = dclab.new_dataset(ddict)
    a = ds.get_kde_scatter(yscale="linear")
    assert np.all(a[:20] == a[0])


def test_kde_log_contour():
    ddict = example_data_dict(size=300, keys=["area_um", "deform"])
    ddict["deform"][:20] = .1
    ddict["area_um"][:20] = .5
    ds = dclab.new_dataset(ddict)
    xm, ym, _ = ds.get_kde_contour(yscale="log")
    dx = np.diff(xm[0])
    dy = np.diff(np.log(ym[:, 0]))
    assert np.allclose(dx, dx[0])
    assert np.allclose(dy, dy[0])


def test_kde_log_scatter():
    ddict = example_data_dict(size=300, keys=["area_um", "deform"])
    ddict["deform"][:20] = .1
    ddict["area_um"][:20] = .5
    ds = dclab.new_dataset(ddict)
    a = ds.get_kde_scatter(yscale="log")
    assert np.all(a[:20] == a[0])


def test_kde_log_scatter_points():
    ddict = example_data_dict(size=300, keys=["area_um", "tilt"])
    ds = dclab.new_dataset(ddict)
    a = ds.get_kde_scatter(yscale="log", xax="area_um", yax="tilt")
    b = ds.get_kde_scatter(yscale="log", xax="area_um", yax="tilt",
                           positions=[ds["area_um"], ds["tilt"]])

    assert np.all(a == b)


def test_kde_log_scatter_invalid():
    ddict = example_data_dict(size=300, keys=["area_um", "deform"])
    ddict["deform"][:20] = .1
    ddict["area_um"][:20] = .5
    ddict["deform"][21] = np.nan
    ddict["deform"][22] = np.inf
    ddict["deform"][23] = -.1
    ds = dclab.new_dataset(ddict)
    a = ds.get_kde_scatter(yscale="log")
    assert np.all(a[:20] == a[0])
    assert np.isnan(a[21])
    assert np.isnan(a[22])
    assert np.isnan(a[23])


def test_kde_none():
    ddict = example_data_dict()
    ds = dclab.new_dataset(ddict)
    sc = ds.get_kde_scatter(kde_type="none")
    assert np.sum(sc) == sc.shape[0]
    ds.get_kde_contour()


def test_kde_nofilt():
    ddict = example_data_dict()
    ds = dclab.new_dataset(ddict)
    ds.config["filtering"]["enable filters"] = False
    sc = ds.get_kde_scatter()
    cc = ds.get_kde_contour()
    assert sc.shape[0] == 100
    # This will fail if the default contour accuracy is changed
    # in `get_kde_contour`.
    assert cc[0].shape == (43, 41)


def test_kde_positions():
    ddict = example_data_dict()
    ds = dclab.new_dataset(ddict)

    ds.config["filtering"]["enable filters"] = False
    sc = ds.get_kde_scatter(xax="area_um", yax="deform")
    sc2 = ds.get_kde_scatter(xax="area_um", yax="deform",
                             positions=(ds["area_um"], ds["deform"]))
    assert np.all(sc == sc2)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
