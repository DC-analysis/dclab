"""Computation of volume for RT-DC measurements based on a rotation
of the contours"""
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
    pix: float
        The detector pixel size in µm.
        e.g. obtained using: `mm.config["imaging"]["pixel size"]`

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
            # Last point of the contour has to overlap with the first point
            z_vec = np.hstack([contour_x, contour_x[0]])

            # Last point of the contour has to overlap with the first point
            contour_y_low = np.hstack([contour_y_low, contour_y_low[0]])
            contour_y_upp = np.hstack([contour_y_upp, contour_y_upp[0]])

            vol_low = vol_revolve(contour_y_low, z_vec, pix)
            vol_upp = vol_revolve(contour_y_upp, z_vec, pix)

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


def vol_revolve(r, z, point_scale=1):
    """Calculate the volume of a polygon revolved around the Z-axis

    Implementation of the volRevolve function (2012-05-03) by Geoff Olynyk
    https://de.mathworks.com/matlabcentral/fileexchange/36525-volrevolve

    Copyright (c) 2012, Geoff Olynyk
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
    TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
    OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
    OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Parameters
    ----------
    r: 1d np.ndarray
        coordinate perpendicular to the axis of rotation
    z: 1d np.ndarray
        coordinate along the axis of rotation
    point_scale: float
        point size in your preferred units (not part of the original
        function); The volume is multiplied by a factor of
        `point_scale**3`.

    Notes
    -----
    - R and Z vectors must be in order, counter-clockwise around the area
      being defined. If not, this will give the volume of the
      counter-clockwise parts, minus the volume of the clockwise parts.

    - It does not matter if the curve is open or closed - if it is open
      (last point doesn't overlap first point), this function will
      automatically close it.

    - Based on Advanced Mathematics and Mechanics Applications with MATLAB,
      3rd ed., by H.B. Wilson, L.H. Turcotte, and D. Halpern,
      Chapman & Hall / CRC Press, 2002, e-ISBN 978-1-4200-3544-5.
      See Chapter 5, Section 5.4, doi: 10.1201/9781420035445.ch5
    """
    # sanity checks
    assert len(r) == len(z)
    assert len(r) >= 3
    assert len(r.shape) == len(z.shape) == 1

    # Instead of x and y, describe the contour by a Radius vector r_vec and y
    # The Contour will be rotated around the x-axis. Therefore it is
    # Important that the Contour has been shifted onto the x-Axis
    z_vec_m1 = z[:-1]
    d_z = z[1:] - z_vec_m1
    r_vec = np.sqrt(z ** 2 + r ** 2)
    rvec_m1 = r_vec[:-1]
    d_r = r_vec[1:] - rvec_m1
    # 4 volume parts
    v1 = d_r * d_z * rvec_m1
    v2 = 2 * d_z * rvec_m1**2
    v3 = -1 * d_r**2 * d_z
    v4 = -2 * d_r * rvec_m1 * z_vec_m1

    v_vec = (np.pi/3) * (v1 + v2 + v3 + v4)
    vol = np.sum(v_vec) * point_scale ** 3
    return abs(vol)
