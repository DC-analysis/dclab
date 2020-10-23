"""
Module of kernels that are able to handle continuous as well as categorical
variables (both ordered and unordered).

This is a slight deviation from the current approach in
statsmodels.nonparametric.kernels where each kernel is a class object.

Having kernel functions rather than classes makes extension to a multivariate
kernel density estimation much easier.

NOTE: As it is, this module does not interact with the existing API
"""


import numpy as np


def gaussian(h, Xi, x):
    """
    Gaussian Kernel for continuous variables
    Parameters
    ----------
    h : 1-D ndarray, shape (K,)
        The bandwidths used to estimate the value of the kernel function.
    Xi : 1-D ndarray, shape (K,)
        The value of the training set.
    x : 1-D ndarray, shape (K,)
        The value at which the kernel density is being estimated.

    Returns
    -------
    kernel_value : ndarray, shape (nobs, K)
        The value of the kernel function at each training point for each var.

    """
    return (1. / np.sqrt(2 * np.pi)) * np.exp(-(Xi - x)**2 / (h**2 * 2.))
