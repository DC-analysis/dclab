#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Crosstalk-correction for fluorescence data"""
from __future__ import division, print_function, unicode_literals

import numpy as np


def get_inversion_matrix(ct21, ct31, ct12, ct32, ct13, ct23):
    """Compute crosstalk inversion matrix

    The crosstalk matrix is

    | c11 c12 c13 |
    | c21 c22 c23 |
    | c31 c32 c33 |

    The diagonal elements are computed from the given crosstalk
    matrix elements such that every row sums up to 1, i.e.

    ct11 = 1 - ct12 - ct13
    ct22 = 1 - ct21 - ct23
    ct33 = 1 - ct31 - ct32

    Parameters
    ----------
    cij: float
        Crosstalk (bleed-through) from channel i to channel j

    Returns
    -------
    inv: np.matrix
        The inverted crosstalk matrix
    """
    ct11 = 1 - ct12 - ct13
    ct22 = 1 - ct21 - ct23
    ct33 = 1 - ct31 - ct32

    if ct11 < 0:
        msg = "ct21+ct31 ({}+{}) must not exceed 1!".format(ct21, ct31)
        raise ValueError(msg)

    if ct21 < 0:
        raise ValueError("ct21 matrix element must not be negative!")

    if ct31 < 0:
        raise ValueError("ct31 matrix element must not be negative!")

    if ct22 < 0:
        msg = "ct12+ct32 ({}+{}) must not exceed 1!".format(ct12, ct32)
        raise ValueError(msg)

    if ct12 < 0:
        raise ValueError("ct12 matrix element must not be negative!")

    if ct32 < 0:
        raise ValueError("ct32 matrix element must not be negative!")

    if ct33 < 0:
        msg = "ct13+ct23 ({}+{}) must not exceed 1!".format(ct13, ct23)
        raise ValueError(msg)

    if ct13 < 0:
        raise ValueError("ct13 matrix element must not be negative!")

    if ct23 < 0:
        raise ValueError("ct23 matrix element must not be negative!")

    crosstalk = np.matrix([[ct11, ct12, ct13],
                           [ct21, ct22, ct23],
                           [ct31, ct32, ct33],
                           ])
    return crosstalk.getI()


def correct_crosstalk(fl1, fl2, fl3, fl_channel,
                      ct21=0, ct31=0, ct12=0, ct32=0, ct13=0, ct23=0):
    """Perform crosstalk correction

    Parameters
    ----------
    fli: int, float, or np.ndarray
        Measured fluorescence signals
    fl_channel: int (1, 2, or 3)
        The channel number for which the crosstalk-corrected signal
        should be computed
    cij: float
        Crosstalk (bleed-through) from channel i to channel j

    See Also
    --------
    get_inversion_matrix: compute the inverse crosstalk matrix

    Notes
    -----
    If there are only two channels (e.g. fl1 and fl2), then the
    crosstalk to and from the other channel (ct31, ct32, ct13, ct23)
    should be set to zero.
    """
    fl_channel = int(fl_channel)
    if fl_channel not in [1, 2, 3]:
        raise ValueError("`fl_channel` must be 1, 2, or 3!")

    minv = get_inversion_matrix(ct21=ct21, ct31=ct31, ct12=ct12,
                                ct32=ct32, ct13=ct13, ct23=ct23)
    
    col = np.array(minv[:, fl_channel - 1]).flatten()
    flout = col[0] * fl1 + col[1] * fl2 + col[2] * fl3
    return flout
