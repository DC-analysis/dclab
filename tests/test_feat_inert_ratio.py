#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from dclab import new_dataset
from dclab.features import inert_ratio as ir

from helper_methods import retrieve_data, cleanup


def test_inert_ratio_raw():
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_mask_contour.zip"))

    raw = ir.get_inert_ratio_raw(cont=ds["contour"])
    ref = np.array([4.25854232,  1.22342663,  4.64971179,  1.70914857,
                    3.62797492, 1.51502192,  2.74757573,  1.79841136])
    assert np.allclose(ref, raw, rtol=0, atol=5e-9)
    cleanup()


def test_inert_ratio_prnc():
    """Test equivalence of inert_ratio_raw and inert_ratio_prnc"""
    t = np.linspace(0, 2*np.pi, 300)

    x1 = 1.7 * np.cos(t)
    y1 = 1.1 * np.sin(t)
    c1 = np.dstack((x1, y1))[0]

    phi = np.arctan2(y1, x1)
    rho = np.sqrt(x1**2 + y1**2)

    for theta in np.linspace(0, 2*np.pi, 14):  # arbitrary rotation
        for pos_x in np.linspace(-5, 20, 8):  # arbitrary x shift
            for pos_y in np.linspace(-4.6, 17, 4):  # arbitrary y shift
                x2 = rho * np.cos(phi + theta) + pos_x
                y2 = rho * np.sin(phi + theta) + pos_y

                c2 = np.dstack((x2, y2))[0]
                raw = ir.get_inert_ratio_raw(c1)
                prnc = ir.get_inert_ratio_prnc(c2, pos_x=pos_x, pos_y=pos_y)

                assert np.allclose(raw, prnc, rtol=0, atol=1e-14)


def test_tilt():
    t = np.linspace(0, 2*np.pi, 300)

    x1 = 1.7 * np.cos(t)
    y1 = 1.1 * np.sin(t)

    phi = np.arctan2(y1, x1)
    rho = np.sqrt(x1**2 + y1**2)

    for theta in np.linspace(-.3, 2.2*np.pi, 32):  # arbitrary rotation
        x2 = rho * np.cos(phi + theta)
        y2 = rho * np.sin(phi + theta)

        c2 = np.dstack((x2, y2))[0]
        tilt = ir.get_tilt(c2)

        th = np.mod(theta, np.pi)
        if th > np.pi/2:
            th = np.pi - th
        assert np.allclose(tilt, th)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
