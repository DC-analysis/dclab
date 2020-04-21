#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals

import numpy as np

from dclab.polygon_filter import PolygonFilter


def test_square_edges():
    poly = np.array([[0, 0],
                     [0, 1],
                     [1, 1],
                     [1, 0]])
    assert PolygonFilter.point_in_poly(p=(0, .5), poly=poly)
    assert not PolygonFilter.point_in_poly(p=(1, .5), poly=poly)
    assert PolygonFilter.point_in_poly(p=(.5, 0), poly=poly)
    assert not PolygonFilter.point_in_poly(p=(.5, 1), poly=poly)


def test_square_corners():
    poly = np.array([[0, 0],
                     [0, 1],
                     [1, 1],
                     [1, 0]])

    assert PolygonFilter.point_in_poly(p=(0, 0), poly=poly)
    assert not PolygonFilter.point_in_poly(p=(1, 0), poly=poly)
    assert not PolygonFilter.point_in_poly(p=(0, 1), poly=poly)
    assert not PolygonFilter.point_in_poly(p=(1, 1), poly=poly)


def test_modsquare_edges_1():
    poly = np.array([[0, 0],
                     [0, 1],
                     [-1, 2],  # a dent to the top left
                     [1, 1],
                     [1, 0]])

    assert PolygonFilter.point_in_poly(p=(0, .5), poly=poly)
    assert not PolygonFilter.point_in_poly(p=(1, .5), poly=poly)
    assert PolygonFilter.point_in_poly(p=(.5, 0), poly=poly)
    assert PolygonFilter.point_in_poly(p=(.5, 1), poly=poly)


def test_modsquare_edges_2():
    poly = np.array([[0, 0],
                     [0, 1],
                     [1, 1],
                     [2, -1],  # a dent to the bottom right
                     [1, 0],
                     ])

    assert PolygonFilter.point_in_poly(p=(0, .5), poly=poly)
    assert PolygonFilter.point_in_poly(p=(1, .5), poly=poly)
    assert PolygonFilter.point_in_poly(p=(.5, 0), poly=poly)
    assert not PolygonFilter.point_in_poly(p=(.5, 1), poly=poly)


def test_modsquare_outside():
    poly = np.array([[0, 0],
                     [0, 1],
                     [-1, 2],  # a dent to the top left
                     [1, 1],
                     [1, 0]])

    assert PolygonFilter.point_in_poly(p=(.5, .5), poly=poly)
    assert not PolygonFilter.point_in_poly(p=(-.5, .5), poly=poly)


def test_triangle_edges():
    poly = np.array([[0, 0],
                     [0, 1],
                     [1, 1],
                     ])

    assert PolygonFilter.point_in_poly(p=(0, .5), poly=poly)
    assert not PolygonFilter.point_in_poly(p=(1, .5), poly=poly)
    assert not PolygonFilter.point_in_poly(p=(.5, 0), poly=poly)
    assert not PolygonFilter.point_in_poly(p=(.5, 1), poly=poly)


if __name__ == "__main__":
    test_modsquare_outside()
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
