"""Kernel Density Estimation methods"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.stats import gaussian_kde, skew

from .cached import Cache
from .external.statsmodels.nonparametric.kernel_density import KDEMultivariate


def bin_num_doane(a):
    """Compute number of bins based on Doane's formula

    Notes
    -----
    If the bin width cannot be determined, then a bin
    number of 5 is returned.

    See Also
    --------
    bin_width_doane: method used to compute the bin width
    """
    bad = np.isnan(a) | np.isinf(a)
    data = a[~bad]
    acc = bin_width_doane(a)
    if acc == 0 or np.isnan(acc):
        num = 5
    else:
        num = int(np.round((data.max() - data.min()) / acc))
    return num


def bin_width_doane(a):
    """Compute contour spacing based on Doane's formula

    References
    ----------
    - `<https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width>`_
    - `<https://stats.stackexchange.com/questions/55134/
      doanes-formula-for-histogram-binning>`_

    Notes
    -----
    Doane's formula is actually designed for histograms. This
    function is kept here for backwards-compatibility reasons.
    It is highly recommended to use :func:`bin_width_percentile`
    instead.
    """
    bad = np.isnan(a) | np.isinf(a)
    data = a[~bad]
    n = data.size
    g1 = skew(data)
    sigma_g1 = np.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
    k = 1 + np.log2(n) + np.log2(1 + np.abs(g1) / sigma_g1)
    acc = (data.max() - data.min()) / k
    return acc


def bin_width_percentile(a):
    """Compute contour spacing based on data percentiles

    The 10th and the 90th percentile of the input data are taken.
    The spacing then computes to the difference between those
    two percentiles divided by 23.

    Notes
    -----
    The Freedman–Diaconis rule uses the interquartile range and
    normalizes to the third root of len(a). Such things do not
    work very well for RT-DC data, because len(a) is huge. Here
    we use just the top and bottom 10th percentiles with a fixed
    normalization.
    """
    bad = np.isnan(a) | np.isinf(a)
    data = a[~bad]
    start = np.percentile(data, 10)
    end = np.percentile(data, 90)
    acc = (end - start) / 23
    return acc


def get_bad_vals(x, y):
    return np.isnan(x) | np.isinf(x) | np.isnan(y) | np.isinf(y)


def ignore_nan_inf(kde_method):
    """Ignores nans and infs from the input data

    Invalid positions in the resulting density are set to nan.
    """
    def new_kde_method(events_x, events_y, xout=None, yout=None,
                       *args, **kwargs):
        bad_in = get_bad_vals(events_x, events_y)
        if xout is None:
            density = np.zeros_like(events_x, dtype=float)
            bad_out = bad_in
            xo = yo = None
        else:
            density = np.zeros_like(xout, dtype=float)
            bad_out = get_bad_vals(xout, yout)
            xo = xout[~bad_out]
            yo = yout[~bad_out]
        # Filter events
        ev_x = events_x[~bad_in]
        ev_y = events_y[~bad_in]
        density[~bad_out] = kde_method(ev_x, ev_y,
                                       xo, yo,
                                       *args, **kwargs)
        density[bad_out] = np.nan
        return density

    doc_add = "\n    Notes\n" +\
              "    -----\n" +\
              "    This is a wrapped version that ignores nan and inf values."
    new_kde_method.__doc__ = kde_method.__doc__ + doc_add

    return new_kde_method


@ignore_nan_inf
@Cache
def kde_gauss(events_x, events_y, xout=None, yout=None):
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
    valid_combi = ((xout is None and yout is None) or
                   (xout is not None and yout is not None)
                   )
    if not valid_combi:
        raise ValueError("Both `xout` and `yout` must be (un)set.")

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


@ignore_nan_inf
@Cache
def kde_histogram(events_x, events_y, xout=None, yout=None, bins=None):
    """ Histogram-based Kernel Density Estimation

    Parameters
    ----------
    events_x, events_y: 1D ndarray
        The input points for kernel density estimation. Input
        is flattened automatically.
    xout, yout: ndarray
        The coordinates at which the KDE should be computed.
        If set to none, input coordinates are used.
    bins: tuple (binsx, binsy)
        The number of bins to use for the histogram.

    Returns
    -------
    density: ndarray, same shape as `xout`
        The KDE for the points in (xout, yout)

    See Also
    --------
    `numpy.histogram2d`
    `scipy.interpolate.RectBivariateSpline`
    """
    valid_combi = ((xout is None and yout is None) or
                   (xout is not None and yout is not None)
                   )
    if not valid_combi:
        raise ValueError("Both `xout` and `yout` must be (un)set.")

    if yout is None and yout is None:
        xout = events_x
        yout = events_y

    if bins is None:
        bins = (max(5, bin_num_doane(events_x)),
                max(5, bin_num_doane(events_y)))

    # Compute the histogram
    hist2d, xedges, yedges = np.histogram2d(x=events_x,
                                            y=events_y,
                                            bins=bins,
                                            normed=True)
    xip = xedges[1:]-(xedges[1]-xedges[0])/2
    yip = yedges[1:]-(yedges[1]-yedges[0])/2

    estimator = RectBivariateSpline(x=xip, y=yip, z=hist2d)
    density = estimator.ev(xout, yout)
    density[density < 0] = 0

    return density.reshape(xout.shape)


def kde_none(events_x, events_y, xout=None, yout=None):
    """No Kernel Density Estimation

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
    valid_combi = ((xout is None and yout is None) or
                   (xout is not None and yout is not None)
                   )
    if not valid_combi:
        raise ValueError("Both `xout` and `yout` must be (un)set.")

    if yout is None and yout is None:
        xout = events_x
        _ = events_y

    return np.ones(xout.shape)


@ignore_nan_inf
@Cache
def kde_multivariate(events_x, events_y, xout=None, yout=None, bw=None):
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
    valid_combi = ((xout is None and yout is None) or
                   (xout is not None and yout is not None)
                   )
    if not valid_combi:
        raise ValueError("Both `xout` and `yout` must be (un)set.")

    if yout is None and yout is None:
        xout = events_x
        yout = events_y
    if bw is None:
        # divide by 2 to make it comparable to histogram KDE
        bw = (bin_width_doane(events_x) / 2,
              bin_width_doane(events_y) / 2)

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
