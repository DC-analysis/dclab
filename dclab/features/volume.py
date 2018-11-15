#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Computation of volume for RT-DC measurements based on a rotation
of the contours"""
from __future__ import division, print_function, unicode_literals
import numpy as np


def get_volume(cont, pos_x, pos_y, pix):
    """Calculate the volume of a polygon revolved around an axis

    The volume estimation assumes rotational symmetry.
    Green`s theorem and the Gaussian divergence theorem allow to
    formulate the volume as a line integral.

    Parameters
    ----------
    cont: ndarray or list of ndarrays of shape (N,2)
        A 2D array that holds the contour of an event [px]
        e.g. obtained using `mm.contour` where  `mm` is an instance
        of `RTDCBase`. The first and second columns of `cont`
        correspond to the x- and y-coordinates of the contour.
    pos_x: float or ndarray of length N
        The x coordinate(s) of the centroid of the event(s) [µm]
        e.g. obtained using `mm.pos_x`
    pos_y: float or ndarray of length N
        The y coordinate(s) of the centroid of the event(s) [µm]
        e.g. obtained using `mm.pos_y`
    px_um: float
        The detector pixel size in µm.
        e.g. obtained using: `mm.config["image"]["pix size"]`

    Returns
    -------
    volume: float or ndarray
        volume in um^3

    Notes
    -----
    The computation of the volume is based on a full rotation of the
    upper and the lower halves of the contour from which the
    average is then used.

    The volume is computed radially from the the center position
    given by (`pos_x`, `pos_y`). For sufficiently smooth contours,
    such as densely sampled ellipses, the center position does not
    play an important role. For contours that are given on a coarse
    grid, as is the case for RT-DC, the center position must be
    given.

    References
    ----------
    - Halpern et al. :cite:`Halpern2002`, chapter 5, Section 5.4
    - This is a translation from a `Matlab script
      <http://de.mathworks.com/matlabcentral/fileexchange/36525-volrevolve>`_
      by Geoff Olynyk.
    """
    if np.isscalar(pos_x):
        cont = [cont]
        ret_list = False
    else:
        ret_list = True

    # Convert input to 1D arrays
    pos_x = np.atleast_1d(pos_x)
    pos_y = np.atleast_1d(pos_y)

    if pos_x.size != pos_y.size:
        raise ValueError("Size of `pos_x` and `pos_y` must match!")

    if pos_x.size > 1 and len(cont) <= 1:
        raise ValueError("Number of given contours too small!")

    # results are stored in a separate array initialized with nans
    v_avg = np.zeros_like(pos_x, dtype=float)*np.nan

    # v_avg has the shape of `pos_x`. We are iterating over the smallest
    # length for `cont` and `pos_x`.
    for ii in range(min(len(cont), pos_x.shape[0])):
        # If the contour has less than 4 pixels, the computation will fail.
        # In that case, the value np.nan is already assigned.
        cc = cont[ii]
        if cc.shape[0] >= 4:
            # Center contour coordinates with given centroid
            contour_x = cc[:, 0] - pos_x[ii] / pix
            contour_y = cc[:, 1] - pos_y[ii] / pix
            # Make sure contour is counter-clockwise
            contour_x, contour_y = counter_clockwise(contour_x, contour_y)
            # Which points are below the x-axis? (y<0)?
            ind_low = np.where(contour_y < 0)
            # These points will be shifted up to y=0 to build an x-axis
            # (wont contribute to lower volume).
            contour_y_low = np.copy(contour_y)
            contour_y_low[ind_low] = 0
            # Which points are above the x-axis? (y>0)?
            ind_upp = np.where(contour_y > 0)
            # These points will be shifted down to y=0 to build an x-axis
            # (wont contribute to upper volume).
            contour_y_upp = np.copy(contour_y)
            contour_y_upp[ind_upp] = 0
            # Move the contour to the left
            Z = contour_x
            # Last point of the contour has to overlap with the first point
            Z = np.hstack([Z, Z[0]])
            Zp = Z[0:-1]
            dZ = Z[1:]-Zp

            # Last point of the contour has to overlap with the first point
            contour_y_low = np.hstack([contour_y_low, contour_y_low[0]])
            contour_y_upp = np.hstack([contour_y_upp, contour_y_upp[0]])

            vol_low = _vol_helper(contour_y_low, Z, Zp, dZ, pix)
            vol_upp = _vol_helper(contour_y_upp, Z, Zp, dZ, pix)

            v_avg[ii] = (vol_low + vol_upp) / 2

    if not ret_list:
        # Do not return a list if the input contour was not in a list
        v_avg = v_avg[0]

    return v_avg


def counter_clockwise(cx, cy):
    """Put contour coordinates into counter-clockwise order

    Parameters
    ----------
    cx, cy: 1d ndarrays
        The x- and y-coordinates of the contour

    Returns
    -------
    cx_cc, cy_cc:
        The x- and y-coordinates of the contour in
        counter-clockwise orientation.
    """
    # test orientation
    angles = np.unwrap(np.arctan2(cy, cx))
    grad = np.gradient(angles)
    if np.average(grad) > 0:
        return cx[::-1], cy[::-1]
    else:
        return cx, cy


def _vol_helper(contour_y, Z, Zp, dZ, pix):
    # Instead of x and y, describe the contour by a Radius vector R and y
    # The Contour will be rotated around the x-axis. Therefore it is
    # Important that the Contour has been shifted onto the x-Axis
    R = np.sqrt(Z**2 + contour_y**2)
    Rp = R[0:-1]
    dR = R[1:] - Rp
    # 4 volume parts
    v1 = dR * dZ * Rp
    v2 = 2 * dZ * Rp**2
    v3 = -1 * dR**2 * dZ
    v4 = -2 * dR * Rp * Zp

    V = (np.pi/3) * (v1 + v2 + v3 + v4)
    vol = np.sum(V) * pix**3
    return abs(vol)
