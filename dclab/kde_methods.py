#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Kernel Density Estimation methods
"""
from __future__ import division, print_function

import numpy as np
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from .cached import Cache


@Cache
def kde_gauss(events_x, events_y, xout=None, yout=None, **kwargs):
    """ Gaussian Kernel Density Estimation
    
    Parameters
    ----------
    events_x, events_y : 1D ndarray
        The input points for kernel density estimation. Input
        is flattened automatically.
    xout, yout : ndarray
        The coordinates at which the KDE should be computed.
        If set to none, input coordinates are used.
    
    Returns
    -------
    density : ndarray, same shape as `xout`
        The KDE for the points in (xout, yout)

    See Also
    --------
    `scipy.stats.gaussian_kde`
    """
    assert (xout is None and yout is None) or (xout is not None and yout is not None)
    if yout is None and yout is None:
        xout = events_x
        yout = events_y
    
    estimator = gaussian_kde([events_x.flatten(), events_y.flatten()])
    density = estimator.evaluate([xout.flatten(), yout.flatten()])

        
    return density.reshape(xout.shape)


def kde_none(events_x, events_y, xout=None, yout=None, **kwargs):
    """ No Kernel Density Estimation
    
    Parameters
    ----------
    events_x, events_y : 1D ndarray
        The input points for kernel density estimation. Input
        is flattened automatically.
    xout, yout : ndarray
        The coordinates at which the KDE should be computed.
        If set to none, input coordinates are used.
    
    Returns
    -------
    density : ndarray, same shape as `xout`
        The KDE for the points in (xout, yout)

    Notes
    -----
    This method is a convenience method that always returns ones in the shape
    that the other methods in this module produce.
    """
    assert (xout is None and yout is None) or (xout is not None and yout is not None)
    if yout is None and yout is None:
        xout = events_x
        yout = events_y
    
    return np.ones(xout.shape)


@Cache
def kde_multivariate(events_x, events_y, bw, xout=None, yout=None, **kwargs):
    """ Multivariate Kernel Density Estimation
    
    Parameters
    ----------
    events_x, events_y : 1D ndarray
        The input points for kernel density estimation. Input
        is flattened automatically.
    bw : tuple (bwx, bwy)
        The bandwith for kernel density estimation.
    xout, yout : ndarray
        The coordinates at which the KDE should be computed.
        If set to none, input coordinates are used.

    Returns
    -------
    density : ndarray, same shape as `xout`
        The KDE for the points in (xout, yout)

    See Also
    --------
    `statsmodels.nonparametric.kernel_density.KDEMultivariate`
    """
    assert (xout is None and yout is None) or (xout is not None and yout is not None)
    if yout is None and yout is None:
        xout = events_x
        yout = events_y
    
    positions = np.vstack([xout.flatten(), yout.flatten()])
    estimator_ly = KDEMultivariate(data=[events_x.flatten(),
                                         events_y.flatten()],
                                   var_type='cc', bw=bw)

    density = estimator_ly.pdf(positions)
    return density.reshape(xout.shape)
    