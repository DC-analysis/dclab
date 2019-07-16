#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Content-based downsampling of ndarrays"""
from __future__ import division, print_function, unicode_literals

import numpy as np

from .cached import Cache


def downsample_rand(a, samples, remove_invalid=False, ret_idx=False):
    """Downsampling by randomly removing points

    Parameters
    ----------
    a: 1d ndarray
        The input array to downsample
    samples: int
        The desired number of samples
    remove_invalid: bool
        Remove nan and inf values before downsampling
    ret_idx: bool
        Also return a boolean array that corresponds to the
        downsampled indices in `a`.

    Returns
    -------
    dsa: 1d ndarray of size `samples`
        The pseudo-randomly downsampled array `a`
    idx: 1d boolean array with same shape as `a`
        Only returned if `ret_idx` is True.
        A boolean array such that `a[idx] == dsa`
    """
    # fixed random state for this method
    rs = np.random.RandomState(seed=47).get_state()
    np.random.set_state(rs)

    samples = int(samples)

    if remove_invalid:
        # slice out nans and infs
        bad = np.isnan(a) | np.isinf(a)
        pool = a[~bad]
    else:
        pool = a

    if samples and (samples < pool.shape[0]):
        keep = np.zeros_like(pool, dtype=bool)
        keep_ids = np.random.choice(np.arange(pool.size),
                                    size=samples,
                                    replace=False)
        keep[keep_ids] = True
        dsa = pool[keep]
    else:
        keep = np.ones_like(pool, dtype=bool)
        dsa = pool

    if remove_invalid:
        # translate the kept values back to the original array
        idx = np.zeros(a.size, dtype=bool)
        idx[~bad] = keep
    else:
        idx = keep

    if ret_idx:
        return dsa, idx
    else:
        return dsa


@Cache
def downsample_grid(a, b, samples, remove_invalid=False, ret_idx=False):
    """Content-based downsampling for faster visualization

    The arrays `a` and `b` make up a 2D scatter plot with high
    and low density values. This method takes out points at
    indices with high density.

    Parameters
    ----------
    a, b: 1d ndarrays
        The input arrays to downsample
    samples: int
        The desired number of samples
    remove_invalid: bool
        Remove nan and inf values before downsampling; if set to
        `True`, the actual number of samples returned might be
        smaller than `samples` due to infinite or nan values.
    ret_idx: bool
        Also return a boolean array that corresponds to the
        downsampled indices in `a` and `b`.

    Returns
    -------
    dsa, dsb: 1d ndarrays of shape (samples,)
        The arrays `a` and `b` downsampled by evenly selecting
        points and pseudo-randomly adding or removing points
        to match `samples`.
    idx: 1d boolean array with same shape as `a`
        Only returned if `ret_idx` is True.
        A boolean array such that `a[idx] == dsa`
    """
    # fixed random state for this method
    rs = np.random.RandomState(seed=47).get_state()

    if remove_invalid:
        # Remove nan and inf values straight from the beginning.
        # This might results in arrays smaller than `samples`,
        # but it makes sure that no inf/nan values will be plotted.
        bad = np.isnan(a) | np.isinf(a) | np.isnan(b) | np.isinf(b)
        ad = a[~bad]
        bd = b[~bad]
    else:
        bad = np.zeros_like(a, dtype=bool)
        ad = a
        bd = b

    keep = np.ones_like(a, dtype=bool)
    keep[bad] = False

    samples = int(samples)

    if samples and samples < ad.size:
        # The events to keep
        keepd = np.zeros_like(ad, dtype=bool)

        # 1. Produce evenly distributed samples
        # Choosing grid-size:
        # - large numbers tend to show actual structures of the sample,
        #   which is not desired for plotting
        # - small numbers tend will not result in too few samples and,
        #   in order to reach the desired samples, the data must be
        #   upsampled again.
        # 300 is about the size of the plot in marker sizes and yields
        # good results.
        grid_size = 300
        xpx = norm(ad, ad, bd) * grid_size
        ypx = norm(bd, bd, ad) * grid_size
        # The events on the grid to process
        toproc = np.ones((grid_size, grid_size), dtype=bool)

        for ii in range(xpx.size):
            xi = xpx[ii]
            yi = ypx[ii]
            # filter for overlapping events
            # Note that `valid` is used here to promote only valid
            # events in this step. However, in step 2, invalid events
            # could be added back. To avoid this scenario, the
            # parameter `remove_invalid` should be set to True.
            if valid(xi, yi) and toproc[int(xi-1), int(yi-1)]:
                toproc[int(xi-1), int(yi-1)] = False
                # include event
                keepd[ii] = True

        # 2. Make sure that we reach `samples` by adding or
        # removing events.
        diff = np.sum(keepd) - samples
        if diff > 0:
            # Too many samples
            rem_indices = np.where(keepd)[0]
            np.random.set_state(rs)
            rem = np.random.choice(rem_indices,
                                   size=diff,
                                   replace=False)
            keepd[rem] = False
        elif diff < 0:
            # Not enough samples
            add_indices = np.where(~keepd)[0]
            np.random.set_state(rs)
            add = np.random.choice(add_indices,
                                   size=abs(diff),
                                   replace=False)
            keepd[add] = True

        assert np.sum(keepd) == samples, "sanity check"
        asd = ad[keepd]
        bsd = bd[keepd]
        assert np.allclose(ad[keepd], asd, equal_nan=True), "sanity check"
        assert np.allclose(bd[keepd], bsd, equal_nan=True), "sanity check"
        keep[~bad] = keepd
    else:
        asd = ad
        bsd = bd

    if ret_idx:
        return asd, bsd, keep
    else:
        return asd, bsd


def valid(a, b):
    """Check whether `a` and `b` are not inf or nan"""
    return ~(np.isnan(a) | np.isinf(a) | np.isnan(b) | np.isinf(b))


def norm(a, ref1, ref2):
    """
    Normalize `a` with min/max values of `ref1`, using all elements of
    `ref1` where the `ref1` and `ref2` are not nan or inf"""
    ref = ref1[valid(ref1, ref2)]
    return (a-ref.min())/(ref.max()-ref.min())
