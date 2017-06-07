#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Content-based downsampling of ndarrays"""
from __future__ import division, print_function, unicode_literals

import numpy as np

from .cached import Cache


def downsample_rand(a, samples, remove_invalid=True, retidx=False):
    """Downsampling by randomly removing points
    
    Parameters
    ----------
    a: 1d ndarray
        The input array to downsample
    samples: int
        The desired number of samples
    remove_invalid: bool
        Remove nan and inf values before downsampling
    retidx: bool
        Also return a boolean array that corresponds to the
        downsampled indices in `a`.

    Returns
    -------
    dsa, dsb: 1d ndarrays of shape (samples,)
        The pseudo-randomly downsampled arrays `a` and `b`
    [idx]: 1d boolean array with same shape as `a`
        A boolean array such that `a[idx] == dsa` is all true
    """
    # fixed random state for this method
    rs = np.random.RandomState(seed=47).get_state()
    
    assert remove_invalid, "Downsampling cannot handle inf/nan yet!"
    
    samples = int(samples)
    
    if remove_invalid:
        # slice out nans and infs
        bad = np.isnan(a)+np.isinf(a)
        a = a[~bad]
    
    if samples and (samples < a.shape[0]):
        keep = np.zeros_like(a, dtype=bool)
        np.random.set_state(rs)
        keep_ids = np.random.choice(np.arange(a.shape[0]),
                                    size=samples,
                                    replace=False)
        keep[keep_ids] = True
        dsa = a[keep]
    else:
        keep = np.ones_like(a, dtype=bool)
        dsa = a

    if remove_invalid:
        # translate the kept values back to the original array
        keep_inv = np.zeros_like(bad)
        keep_inv[~bad] = keep

    if retidx:
        return dsa, keep_inv
    else:
        return dsa


@Cache
def downsample_grid(a, b, samples, remove_invalid=True, retidx=False):
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
        Remove nan and inf values before downsampling
    retidx: bool
        Also return a boolean array that corresponds to the
        downsampled indices in `a` and `b`.

    Returns
    -------
    dsa, dsb: 1d ndarrays of shape (samples,)
        The arrays `a` and `b` downsampled by evenly selecting
        points and pseudo-randomly adding or removing points
        to match `samples`.
    [idx]: 1d boolean array with same shape as `a`
        A boolean array such that `a[idx] == dsa` is all true

    """
    # fixed random state for this method
    rs = np.random.RandomState(seed=47).get_state()
    
    assert remove_invalid, "Downsampling cannot handle inf/nan yet!"
    
    samples = int(samples)
    
    if remove_invalid:
        # slice out nans and infs
        bad = np.isnan(a)+np.isinf(a)+np.isnan(b)+np.isinf(b)
        a = a[~bad]
        b = b[~bad]

    if samples and samples < a.shape[0]:
        # The events to keep
        keep = np.zeros(a.shape, dtype=bool)
    
        # 1. Produce evenly distributed samples
        # Choosing grid-size:
        # - large numbers tend to show actual structures of the sample,
        #   which is not desired for plotting
        # - small numbers tend will not result in too few samples and,
        #   in order to reach the desired samples, the data must be
        #   upsampled again.
        # 300 is about the size of the plot in marker sizes and yields
        # good results.
        grid_size=300
        xpx = (a-a.min())/(a.max()-a.min()) * grid_size
        ypx = (b-b.min())/(b.max()-b.min()) * grid_size    
        # The events on the grid to process
        toproc = np.ones((grid_size, grid_size), dtype=bool)
    
        for ii in range(xpx.shape[0]):
            xi = xpx[ii]
            yi = ypx[ii]
            ## first filter for exactly overlapping events
            if toproc[int(xi-1), int(yi-1)]:
                toproc[int(xi-1), int(yi-1)] = False
                ## second filter for multiple overlay
                keep[ii] = True

        
        # 2. Make sure that we reach `samples` by adding or
        # removing events.        
        diff = np.sum(keep) - samples
        if diff > 0:
            # Too many samples
            rem_indices = np.where(keep==True)[0]
            np.random.set_state(rs)
            rem = np.random.choice(rem_indices,
                                   size=diff,
                                   replace=False)
            keep[rem] = False
        elif diff < 0:
            # Not enough samples
            add_indices = np.where(keep==False)[0]
            np.random.set_state(rs)
            add = np.random.choice(add_indices,
                                   size=abs(diff),
                                   replace=False)
            keep[add] = True    

        assert np.sum(keep) == samples    
        asd = a[keep]
        bsd = b[keep]
    else:
        keep = np.ones_like(a, dtype=bool)
        asd = a
        bsd = b

    if remove_invalid:
        # translate the kept values back to the original array
        keep_inv = np.zeros_like(bad)
        keep_inv[~bad] = keep
        
    if retidx:
        return asd, bsd, keep_inv
    else:
        return asd, bsd
