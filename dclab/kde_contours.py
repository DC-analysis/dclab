from __future__ import division

import numpy as np

from .external.skimage.measure import find_contours, points_in_poly
import scipy.interpolate as spint

from .kde_methods import get_bad_vals


def find_contours_level(density, x, y, level, closed=False):
    """Find iso-valued density contours for a given level value

    Parameters
    ----------
    density: 2d ndarray of shape (M, N)
        Kernel density estimate for which to compute the contours
    x: 2d ndarray of shape (M, N) or 1d ndarray of size M
        X-values corresponding to `density`
    y: 2d ndarray of shape (M, N) or 1d ndarray of size M
        Y-values corresponding to `density`
    level: float between 0 and 1
        Value along which to find contours in `density` relative
        to its maximum

    Returns
    -------
    contours: list of ndarrays of shape (P, 2)
        Contours found for the given level value

    See Also
    --------
    skimage.measure.find_contours: Contour finding algorithm used
    """
    if level >= 1 or level <= 0:
        raise ValueError("`level` must be in (0,1), got '{}'!".format(level))
    # level relative to maximum
    level = level * density.max()
    # xy coordinates
    if len(x.shape) == 2:
        assert np.all(x[:, 0] == x[:, 1])
        x = x[:, 0]
    if len(y.shape) == 2:
        assert np.all(y[0, :] == y[1, :])
        y = y[0, :]
    if closed:
        # find closed contours
        density = np.pad(density, ((1, 1), (1, 1)), mode="constant")
        offset = 1
    else:
        # leave contours open at kde boundary
        offset = 0

    conts_idx = find_contours(density, level)
    conts_xy = []

    for cc in conts_idx:
        cx = np.interp(x=cc[:, 0]-offset,
                       xp=range(x.size),
                       fp=x)
        cy = np.interp(x=cc[:, 1]-offset,
                       xp=range(y.size),
                       fp=y)
        conts_xy.append(np.stack((cx, cy), axis=1))

    return conts_xy


def get_quantile_levels(density, x, y, xp, yp, q, normalize=True):
    """Compute density levels for given quantiles by interpolation

    For a given 2D density, compute the density levels at which
    the resulting contours contain the fraction `1-q` of all
    data points. E.g. for a measurement of 1000 events, all
    contours at the level corresponding to a quantile of
    `q=0.95` (95th percentile) contain 50 events (5%).

    Parameters
    ----------
    density: 2d ndarray of shape (M, N)
        Kernel density estimate for which to compute the contours
    x: 2d ndarray of shape (M, N) or 1d ndarray of size M
        X-values corresponding to `density`
    y: 2d ndarray of shape (M, N) or 1d ndarray of size M
        Y-values corresponding to `density`
    xp: 1d ndarray of size D
        Event x-data from which to compute the quantile
    yp: 1d ndarray of size D
        Event y-data from which to compute the quantile
    q: array_like or float between 0 and 1
        Quantile along which to find contours in `density` relative
        to its maximum
    normalize: bool
        Whether output levels should be normalized to the maximum
        of `density`

    Returns
    -------
    level: np.ndarray or float
        Contours level(s) corresponding to the given quantile

    Notes
    -----
    NaN-values events in `xp` and `yp` are ignored.
    """
    # xy coordinates
    if len(x.shape) == 2:
        assert np.all(x[:, 0] == x[:, 1])
        x = x[:, 0]
    if len(y.shape) == 2:
        assert np.all(y[0, :] == y[1, :])
        y = y[0, :]

    # remove bad events
    bad = get_bad_vals(xp, yp)
    xp = xp[~bad]
    yp = yp[~bad]

    # Normalize interpolation data such that the spacing for
    # x and y is about the same during interpolation.
    x_norm = x.max()
    x = x / x_norm
    xp = xp / x_norm

    y_norm = y.max()
    y = y / y_norm
    yp = yp / y_norm

    # Perform interpolation
    dp = spint.interpn((x, y), density,
                       (xp, yp),
                       method='linear',
                       bounds_error=False,
                       fill_value=0)

    if normalize:
        dp /= density.max()

    if not np.isscalar(q):
        q = np.array(q)
    plev = np.nanpercentile(dp, q=q*100)
    return plev


def _find_quantile_level(density, x, y, xp, yp, quantile, acc=.01,
                         ret_err=False):
    """Find density level for a given data quantile by iteration

    Parameters
    ----------
    density: 2d ndarray of shape (M, N)
        Kernel density estimate for which to compute the contours
    x: 2d ndarray of shape (M, N) or 1d ndarray of size M
        X-values corresponding to `density`
    y: 2d ndarray of shape (M, N) or 1d ndarray of size M
        Y-values corresponding to `density`
    xp: 1d ndarray of size D
        Event x-data from which to compute the quantile
    yp: 1d ndarray of size D
        Event y-data from which to compute the quantile
    quantile: float between 0 and 1
        Quantile along which to find contours in `density` relative
        to its maximum
    acc: float
        Desired absolute accuracy (stopping criterion) of the
        contours
    ret_err: bool
        If True, also return the absolute error

    Returns
    -------
    level: float
        Contours level corresponding to the given quantile

    Notes
    -----
    A much more faster method (using interpolation) is implemented in
    :func:`get_quantile_levels`.
    NaN-values events in `xp` and `yp` are ignored.

    See Also
    --------
    skimage.measure.find_contours: Contour finding algorithm
    """
    if quantile >= 1 or quantile <= 0:
        raise ValueError("Invalid value for `quantile`: {}".format(quantile))

    # remove bad events
    bad = get_bad_vals(xp, yp)
    xp = xp[~bad]
    yp = yp[~bad]
    points = np.concatenate((xp.reshape(-1, 1), yp.reshape(-1, 1)), axis=1)

    # initial guess
    level = quantile
    # error of current iteration
    err = 1
    # iteration factor (guarantees convergence)
    itfac = 1
    # total number of events
    nev = xp.size

    while np.abs(err) > acc:
        # compute contours
        conts = find_contours_level(density, x, y, level, closed=True)
        # compute number of points in contour
        isin = 0
        pi = np.array(points, copy=True)
        for cc in conts:
            pinc = points_in_poly(points=pi, verts=cc)
            isin += np.sum(pinc)
            # ignore these points for the other contours
            pi = pi[~pinc]
        err = quantile - (nev - isin) / nev
        level += err * itfac
        itfac *= .9

    if ret_err:
        return level, err
    else:
        return level
