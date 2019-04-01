#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Computation of event contour from event mask"""
from __future__ import division, print_function, unicode_literals

import numpy as np

# equivalent to
# from skimage.measure import find_contours
from ..external.skimage.measure import find_contours


def get_contour(mask):
    """Compute the image contour from a mask

    The contour is computed in a very inefficient way using scikit-image
    and a conversion of float coordinates to pixel coordinates.

    Parameters
    ----------
    mask: binary ndarray of shape (M,N) or (K,M,N)
        The mask outlining the pixel positions of the event.
        If a 3d array is given, then `K` indexes the individual
        contours.

    Returns
    -------
    cont: ndarray or list of K ndarrays of shape (J,2)
        A 2D array that holds the contour of an event (in pixels)
        e.g. obtained using `mm.contour` where  `mm` is an instance
        of `RTDCBase`. The first and second columns of `cont`
        correspond to the x- and y-coordinates of the contour.
    """
    if isinstance(mask, np.ndarray) and len(mask.shape) == 2:
        mask = [mask]
        ret_list = False
    else:
        ret_list = True
    contours = []

    for mi in mask:
        c0 = find_contours(mi.transpose(),
                           level=.9999,
                           positive_orientation="low",
                           fully_connected="high")[0]
        # round all coordinates to pixel values
        c1 = np.asarray(np.round(c0), int)
        # remove duplicates
        c2 = remove_duplicates(c1)
        contours.append(c2)
    if ret_list:
        return contours
    else:
        return contours[0]


def remove_duplicates(cont):
    out = []
    for ii in range(len(cont)):
        if np.all(cont[ii] == cont[ii - 1]):
            pass
        else:
            out.append(cont[ii])
    return np.array(out)
