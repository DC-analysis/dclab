#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals

import os
import tempfile

import numpy as np
import pytest

import dclab

from helper_methods import example_data_dict


filter_data = """[Polygon 00000000]
X Axis = area_um
Y Axis = deform
Name = polygon filter 0
point00000000 = 6.344607717656481e-03 7.703315881326352e-01
point00000001 = 7.771010748302133e-01 7.452006980802792e-01
point00000002 = 8.025596093384512e-01 6.806282722513089e-03
point00000003 = 6.150993521573982e-01 1.015706806282723e-03
"""


@pytest.mark.filterwarnings('ignore::dclab.polygon_filter.'
                            + 'FilterIdExistsWarning')
def test_import():
    dclab.PolygonFilter.clear_all_filters()
    ddict = example_data_dict(size=1000, keys=["area_um", "deform"])
    ds = dclab.new_dataset(ddict)

    # save polygon data
    _fd, tf = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf, "w") as fd:
        fd.write(filter_data)

    # Add polygon filter
    pf = dclab.PolygonFilter(filename=tf)
    ds.polygon_filter_add(pf)

    ds.apply_filter()

    assert np.sum(ds._filter) == 330

    dclab.PolygonFilter.import_all(tf)

    assert len(dclab.PolygonFilter.instances) == 2

    # Import multiples
    b = filter_data
    b = b.replace("Polygon 00000000", "Polygon 00000001")
    with open(tf, "a") as fd:
        fd.write(b)
    dclab.PolygonFilter.import_all(tf)

    # Import previously saved
    dclab.PolygonFilter.save_all(tf)
    dclab.PolygonFilter.import_all(tf)

    assert len(dclab.PolygonFilter.instances) == 10

    try:
        os.remove(tf)
    except OSError:
        pass


def test_invert():
    dclab.PolygonFilter.clear_all_filters()
    ddict = example_data_dict(size=1234, keys=["area_um", "deform"])
    ds = dclab.new_dataset(ddict)
    # points of polygon filter
    points = [[np.min(ddict["area_um"]), np.min(ddict["deform"])],
              [np.min(ddict["area_um"]), np.max(ddict["deform"])],
              [np.average(ddict["area_um"]), np.max(ddict["deform"])],
              [np.average(ddict["area_um"]), np.min(ddict["deform"])],
              ]
    filt1 = dclab.PolygonFilter(axes=["area_um", "deform"],
                                points=points,
                                inverted=False)
    ds.polygon_filter_add(filt1)
    assert [0] == ds.config["filtering"]["polygon filters"]
    n1 = np.sum(ds._filter)
    ds.apply_filter()
    n2 = np.sum(ds._filter)
    assert n1 != n2
    filt2 = dclab.PolygonFilter(axes=["area_um", "deform"],
                                points=points,
                                inverted=True)
    ds.polygon_filter_add(filt2)
    assert [0, 1] == ds.config["filtering"]["polygon filters"]
    ds.apply_filter()
    assert np.sum(ds._filter) == 0, "inverted+normal filter filters all"
    dclab.PolygonFilter.clear_all_filters()


def test_invert_copy():
    dclab.PolygonFilter.clear_all_filters()
    ddict = example_data_dict(size=1234, keys=["area_um", "deform"])
    ds = dclab.new_dataset(ddict)
    # points of polygon filter
    points = [[np.min(ddict["area_um"]), np.min(ddict["deform"])],
              [np.min(ddict["area_um"]), np.max(ddict["deform"])],
              [np.average(ddict["area_um"]), np.max(ddict["deform"])],
              [np.average(ddict["area_um"]), np.min(ddict["deform"])],
              ]
    filt1 = dclab.PolygonFilter(axes=["area_um", "deform"],
                                points=points,
                                inverted=False)
    ds.polygon_filter_add(filt1)
    assert [0] == ds.config["filtering"]["polygon filters"]
    n1 = np.sum(ds._filter)
    ds.apply_filter()
    n2 = np.sum(ds._filter)
    assert n1 != n2
    filt2 = filt1.copy(invert=True)
    ds.polygon_filter_add(filt2)
    assert [0, 1] == ds.config["filtering"]["polygon filters"]
    ds.apply_filter()
    assert np.sum(ds._filter) == 0, "inverted+normal filter filters all"
    dclab.PolygonFilter.clear_all_filters()


@pytest.mark.filterwarnings('ignore::dclab.polygon_filter.'
                            + 'FilterIdExistsWarning')
def test_invert_saveload():
    dclab.PolygonFilter.clear_all_filters()
    ddict = example_data_dict(size=1234, keys=["area_um", "deform"])
    # points of polygon filter
    points = [[np.min(ddict["area_um"]), np.min(ddict["deform"])],
              [np.min(ddict["area_um"]), np.max(ddict["deform"])],
              [np.average(ddict["area_um"]), np.max(ddict["deform"])],
              [np.average(ddict["area_um"]), np.min(ddict["deform"])],
              ]
    filt1 = dclab.PolygonFilter(axes=["area_um", "deform"],
                                points=points,
                                inverted=True)
    name = tempfile.mktemp(prefix="test_dclab_polygon_")
    filt1.save(name)
    filt2 = dclab.PolygonFilter(filename=name)
    assert filt2 == filt1

    filt3 = dclab.PolygonFilter(axes=["area_um", "deform"],
                                points=points,
                                inverted=False)
    try:
        os.remove(name)
    except OSError:
        pass

    name = tempfile.mktemp(prefix="test_dclab_polygon_")
    filt3.save(name)
    filt4 = dclab.PolygonFilter(filename=name)
    assert filt4 == filt3
    try:
        os.remove(name)
    except OSError:
        pass


def test_inverted_wrong():
    dclab.PolygonFilter.clear_all_filters()
    ddict = example_data_dict(size=1234, keys=["area_um", "deform"])
    # points of polygon filter
    points = [[np.min(ddict["area_um"]), np.min(ddict["deform"])],
              [np.min(ddict["area_um"]), np.max(ddict["deform"])],
              [np.average(ddict["area_um"]), np.max(ddict["deform"])],
              [np.average(ddict["area_um"]), np.min(ddict["deform"])],
              ]
    try:
        dclab.PolygonFilter(axes=["area_um", "deform"],
                            points=points,
                            inverted=0)
    except dclab.polygon_filter.PolygonFilterError:
        pass
    else:
        raise ValueError("inverted should only be allowed to be bool")


def test_nofile_copy():
    dclab.PolygonFilter.clear_all_filters()
    a = dclab.PolygonFilter(axes=("deform", "area_um"),
                            points=[[0, 1], [1, 1]])
    a.copy()
    dclab.PolygonFilter.clear_all_filters()


def test_remove():
    dclab.PolygonFilter.clear_all_filters()

    _fd, tf = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf, "w") as fd:
        fd.write(filter_data)

    # Add polygon filter
    pf = dclab.PolygonFilter(filename=tf)

    dclab.PolygonFilter.remove(pf.unique_id)
    assert len(dclab.PolygonFilter.instances) == 0

    dclab.PolygonFilter.clear_all_filters()
    try:
        os.remove(tf)
    except OSError:
        pass


@pytest.mark.filterwarnings('ignore::dclab.polygon_filter.'
                            + 'FilterIdExistsWarning')
def test_save():
    dclab.PolygonFilter.clear_all_filters()

    _fd, tf = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf, "w") as fd:
        fd.write(filter_data)

    # Add polygon filter
    pf = dclab.PolygonFilter(filename=tf)

    _fd, tf2 = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf2, "w") as fd:
        fd.write(filter_data)
        pf.save(tf2, ret_fobj=True)
        pf2 = dclab.PolygonFilter(filename=tf2)
        assert np.allclose(pf.points, pf2.points)

    _fd, tf3 = tempfile.mkstemp(prefix="dclab_polgyon_test")
    dclab.PolygonFilter.save_all(tf3)
    pf.save(tf3, ret_fobj=False)

    # ensure backwards compatibility: the names of the
    # three filters should be the same
    names = dclab.polygon_filter.get_polygon_filter_names()
    assert len(names) == 2
    assert names.count(names[0]) == 2

    try:
        os.remove(tf)
        os.remove(tf2)
        os.remove(tf3)
    except OSError:
        pass


@pytest.mark.filterwarnings('ignore::dclab.polygon_filter.'
                            + 'FilterIdExistsWarning')
def test_save_multiple():
    dclab.PolygonFilter.clear_all_filters()

    _fd, tf = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf, "w") as fd:
        fd.write(filter_data)

    # Add polygon filter
    pf = dclab.PolygonFilter(filename=tf)

    _fd, tf2 = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf2, "a") as fd:
        pf.save(fd)
        pf2 = dclab.PolygonFilter(filename=tf2)
        assert np.allclose(pf.points, pf2.points)

    try:
        os.remove(tf)
        os.remove(tf2)
    except OSError:
        pass


@pytest.mark.filterwarnings('ignore::dclab.polygon_filter.'
                            + 'FilterIdExistsWarning')
def test_unique_id():
    dclab.PolygonFilter.clear_all_filters()
    _fd, tf = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf, "w") as fd:
        fd.write(filter_data)

    # Add polygon filter
    pf = dclab.PolygonFilter(filename=tf, unique_id=2)
    pf2 = dclab.PolygonFilter(filename=tf, unique_id=2)
    assert pf.unique_id != pf2.unique_id
    dclab.PolygonFilter.clear_all_filters()

    try:
        os.remove(tf)
    except OSError:
        pass


@pytest.mark.filterwarnings('ignore::dclab.polygon_filter.'
                            + 'FilterIdExistsWarning')
def test_with_rtdc_data_set():
    dclab.PolygonFilter.clear_all_filters()
    ddict = example_data_dict(size=821, keys=["area_um", "deform"])
    ds = dclab.new_dataset(ddict)

    # save polygon data
    _fd, tf = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf, "w") as fd:
        fd.write(filter_data)
    pf = dclab.PolygonFilter(filename=tf)
    pf2 = dclab.PolygonFilter(filename=tf)

    ds.polygon_filter_add(pf)
    ds.polygon_filter_add(1)

    ds.polygon_filter_rm(0)
    ds.polygon_filter_rm(pf2)

    dclab.PolygonFilter.clear_all_filters()
    try:
        os.remove(tf)
    except OSError:
        pass


def test_wrong_load_key():
    dclab.PolygonFilter.clear_all_filters()

    # save polygon data
    _fd, tf = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf, "w") as fd:
        fd.write(filter_data + "peter=4\n")

    try:
        dclab.PolygonFilter(filename=tf)
    except KeyError:
        pass
    else:
        raise ValueError("_load should not accept unknown key!")
    dclab.PolygonFilter.clear_all_filters()
    try:
        os.remove(tf)
    except OSError:
        pass


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
