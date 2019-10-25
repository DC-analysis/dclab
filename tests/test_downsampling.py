#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from dclab import downsampling


def test_basic():
    a = np.arange(100)
    b, idx = downsampling.downsample_rand(a=a,
                                          samples=5,
                                          ret_idx=True)
    assert np.all(a[idx] == b)

    b2 = downsampling.downsample_rand(a=a,
                                      samples=5,
                                      ret_idx=False)
    assert np.all(b2 == b)


def test_grid_basic():
    a = np.arange(100)
    b = np.arange(50, 150)
    ads, bds, idx = downsampling.downsample_grid(a=a,
                                                 b=b,
                                                 samples=5,
                                                 ret_idx=True)
    assert np.all(a[idx] == ads)
    assert np.all(b[idx] == bds)

    ads2, bds2 = downsampling.downsample_grid(a=a,
                                              b=b,
                                              samples=5,
                                              ret_idx=False)
    assert np.all(ads2 == ads)
    assert np.all(bds2 == bds)


def test_grid_nan():
    a = np.arange(100, dtype=float)
    a[50:] = np.nan
    b = np.arange(50, 150, dtype=float)
    ads, bds, idx = downsampling.downsample_grid(a=a,
                                                 b=b,
                                                 samples=50,
                                                 ret_idx=True,
                                                 remove_invalid=False)
    assert np.allclose(a[idx], ads, atol=1e-14, rtol=0, equal_nan=True)
    assert np.allclose(b[idx], bds, atol=1e-14, rtol=0, equal_nan=True)
    assert np.sum(np.isnan(ads)) == 1, "depends on random state"

    ads2, bds2 = downsampling.downsample_grid(a=a,
                                              b=b,
                                              samples=50,
                                              ret_idx=False,
                                              remove_invalid=True)
    assert np.sum(np.isnan(ads2)) == 0
    assert np.all(ads2 == a[:50])
    assert np.all(bds2 == b[:50])

    ads3, bds3, idx3 = downsampling.downsample_grid(a=a,
                                                    b=b,
                                                    samples=60,
                                                    ret_idx=True,
                                                    remove_invalid=True)
    assert np.sum(np.isnan(bds3)) == 0
    assert ads3.size == 50, "only 50 valid values"
    assert np.all(ads3 == a[idx3])
    assert np.all(bds3 == b[idx3])


def test_rand_nan():
    a = np.arange(100, dtype=float)
    a[50:] = np.nan
    b, idx = downsampling.downsample_rand(a=a,
                                          samples=5,
                                          ret_idx=True,
                                          remove_invalid=False)
    assert np.allclose(a[idx], b, atol=1e-14, rtol=0, equal_nan=True)
    assert np.sum(np.isnan(b)) == 4

    b2, idx2 = downsampling.downsample_rand(a=a,
                                            samples=5,
                                            ret_idx=True,
                                            remove_invalid=True)
    assert np.allclose(a[idx2], b2, atol=1e-14, rtol=0, equal_nan=True)
    assert np.sum(np.isnan(b2)) == 0


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
