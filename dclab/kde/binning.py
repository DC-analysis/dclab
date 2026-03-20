import warnings

import numpy as np
from scipy.stats import skew


class KernelDensityEstimationForEmtpyArrayWarning(UserWarning):
    """Used when user attempts to compute KDE for an empty array"""


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
    if n > 0:
        g1 = skew(data)
        sigma_g1 = np.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
        k = 1 + np.log2(n) + np.log2(1 + np.abs(g1) / sigma_g1)
        acc = (data.max() - data.min()) / k
    else:
        warnings.warn("KDE encountered an empty array",
                      KernelDensityEstimationForEmtpyArrayWarning)
        acc = 1
    return acc


def bin_width_doane_div5(a):
    """Compute contour spacing based on Doane's formula divided by five

    See Also
    --------
    bin_width_doane: method used to compute the bin width
    """
    return bin_width_doane(a) / 5


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
    if not data.size:
        warnings.warn("KDE encountered an empty array",
                      KernelDensityEstimationForEmtpyArrayWarning)
        acc = 1
    else:
        start = np.percentile(data, 10)
        end = np.percentile(data, 90)
        acc = (end - start) / 23
    return acc
