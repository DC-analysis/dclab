#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Kernel Density Estimation methods
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from .cached import Cache


@Cache
def kde_gauss(events_x, events_y, xout=None, yout=None, **kwargs):
    """ Gaussian Kernel Density Estimation
    
    Parameters
    ----------
    events_x, events_y: 1D ndarray
        The input points for kernel density estimation. Input
        is flattened automatically.
    xout, yout: ndarray
        The coordinates at which the KDE should be computed.
        If set to none, input coordinates are used.
    
    Returns
    -------
    density: ndarray, same shape as `xout`
        The KDE for the points in (xout, yout)

    See Also
    --------
    `scipy.stats.gaussian_kde`
    """
    assert (xout is None and yout is None) or (xout is not None and yout is not None)
    if yout is None and yout is None:
        xout = events_x
        yout = events_y
    
    try:
        estimator = gaussian_kde([events_x.flatten(), events_y.flatten()])
        density = estimator.evaluate([xout.flatten(), yout.flatten()])
    except np.linalg.LinAlgError:
        # LinAlgError occurs when matrix to solve is singular (issue #117)
        density = np.zeros(xout.shape)*np.nan
       
    return density.reshape(xout.shape)


@Cache
def kde_histogram(events_x, events_y, bins=(47,47), xout=None, yout=None, **kwargs):
    """ Histogram-based Kernel Density Estimation
    
    Parameters
    ----------
    events_x, events_y: 1D ndarray
        The input points for kernel density estimation. Input
        is flattened automatically.
    bins: tuple (binsx, binsy)
        The number of bins to use for the histogram.
    xout, yout: ndarray
        The coordinates at which the KDE should be computed.
        If set to none, input coordinates are used.
    
    Returns
    -------
    density: ndarray, same shape as `xout`
        The KDE for the points in (xout, yout)

    See Also
    --------
    `numpy.histogram2d`
    `scipy.interpolate.RectBivariateSpline`
    """
    assert (xout is None and yout is None) or (xout is not None and yout is not None)
    if yout is None and yout is None:
        xout = events_x
        yout = events_y
    
    # Compute the histogram
    hist2d, xedges, yedges = np.histogram2d(x=events_x,
                                            y=events_y,
                                            bins=bins,
                                            normed=True)
    xip = xedges[1:]-(xedges[1]-xedges[0])/2
    yip = yedges[1:]-(yedges[1]-yedges[0])/2
    
    estimator = RectBivariateSpline(x=xip, y=yip, z=hist2d)
    density = estimator.ev(xout, yout)
    density[density<0] = 0
    
    return density.reshape(xout.shape)


def kde_none(events_x, events_y, xout=None, yout=None, **kwargs):
    """ No Kernel Density Estimation
    
    Parameters
    ----------
    events_x, events_y: 1D ndarray
        The input points for kernel density estimation. Input
        is flattened automatically.
    xout, yout: ndarray
        The coordinates at which the KDE should be computed.
        If set to none, input coordinates are used.
    
    Returns
    -------
    density: ndarray, same shape as `xout`
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
def kde_multivariate(events_x, events_y, bw=None, xout=None, yout=None, **kwargs):
    """ Multivariate Kernel Density Estimation
    
    Parameters
    ----------
    events_x, events_y: 1D ndarray
        The input points for kernel density estimation. Input
        is flattened automatically.
    bw: tuple (bwx, bwy) or None
        The bandwith for kernel density estimation.
    xout, yout: ndarray
        The coordinates at which the KDE should be computed.
        If set to none, input coordinates are used.

    Returns
    -------
    density: ndarray, same shape as `xout`
        The KDE for the points in (xout, yout)

    See Also
    --------
    `statsmodels.nonparametric.kernel_density.KDEMultivariate`
    """
    assert (xout is None and yout is None) or (xout is not None and yout is not None)
    if yout is None and yout is None:
        xout = events_x
        yout = events_y
    if bw is None:
        # Sensible default parameters
        cpstep = lambda a: (np.nanmax(a)-np.nanmin(a))/10
        bw = cpstep(events_x), cpstep(events_y)
    
    positions = np.vstack([xout.flatten(), yout.flatten()])
    estimator_ly = KDEMultivariate(data=[events_x.flatten(),
                                         events_y.flatten()],
                                   var_type='cc', bw=bw)

    density = estimator_ly.pdf(positions)
    return density.reshape(xout.shape)


methods = {"gauss": kde_gauss,
           "histogram": kde_histogram,
           "none": kde_none,
           "multivariate": kde_multivariate}

