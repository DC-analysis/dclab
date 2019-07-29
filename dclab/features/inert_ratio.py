#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Computation of inertia ratio from contour data"""
from __future__ import division, print_function, unicode_literals

import numpy as np
import scipy.spatial as ssp


def cont_moments_cv(cont,
                    flt_epsilon=1.19209e-07,
                    dbl_epsilon=2.2204460492503131e-16):
    """Compute the moments of a contour

    The moments are computed in the same way as they are computed
    in OpenCV's `contourMoments` in `moments.cpp`.

    Parameters
    ----------
    cont: array of shape (N,2)
        The contour for which to compute the moments.
    flt_epsilon: float
        The value of ``FLT_EPSILON`` in OpenCV/gcc.
    dbl_epsilon: float
        The value of ``DBL_EPSILON`` in OpenCV/gcc.

    Returns
    -------
    moments: dict
        A dictionary of moments. If the moment `m00` is smaller
        than half of `flt_epsilon`, `None` is returned.
    """
    # Make sure we have no unsigned integers
    if np.issubdtype(cont.dtype, np.unsignedinteger):
        cont = cont.astype(np.int)

    xi = cont[:, 0]
    yi = cont[:, 1]

    xi_1 = np.roll(xi, -1)
    yi_1 = np.roll(yi, -1)

    xi_12 = xi_1**2
    yi_12 = yi_1**2

    xi2 = xi**2
    yi2 = yi**2

    dxy = xi_1 * yi - xi * yi_1

    xii_1 = xi_1 + xi
    yii_1 = yi_1 + yi

    a00 = np.sum(dxy)
    a10 = np.sum(dxy * xii_1)
    a01 = np.sum(dxy * yii_1)
    a20 = np.sum(dxy * (xi_1 * xii_1 + xi2))
    a11 = np.sum(dxy * (xi_1 * (yii_1 + yi_1) + xi * (yii_1 + yi)))
    a02 = np.sum(dxy * (yi_1 * yii_1 + yi2))
    a30 = np.sum(dxy * xii_1 * (xi_12 + xi2))
    a03 = np.sum(dxy * yii_1 * (yi_12 + yi2))
    a21 = np.sum(dxy * (xi_12 * (3 * yi_1 + yi) + 2 *
                        xi * xi_1 * yii_1 + xi2 * (yi_1 + 3 * yi)))
    a12 = np.sum(dxy * (yi_12 * (3 * xi_1 + xi) + 2 *
                        yi * yi_1 * xii_1 + yi2 * (xi_1 + 3 * xi)))

    if abs(a00) > flt_epsilon:
        db1_2 = 0.5
        db1_6 = 0.16666666666666666666666666666667
        db1_12 = 0.083333333333333333333333333333333
        db1_24 = 0.041666666666666666666666666666667
        db1_20 = 0.05
        db1_60 = 0.016666666666666666666666666666667

        if a00 < 0:
            db1_2 *= -1
            db1_6 *= -1
            db1_12 *= -1
            db1_24 *= -1
            db1_20 *= -1
            db1_60 *= -1

        m = dict(m00=a00 * db1_2,
                 m10=a10 * db1_6,
                 m01=a01 * db1_6,
                 m20=a20 * db1_12,
                 m11=a11 * db1_24,
                 m02=a02 * db1_12,
                 m30=a30 * db1_20,
                 m21=a21 * db1_60,
                 m12=a12 * db1_60,
                 m03=a03 * db1_20,
                 )

        if m["m00"] > dbl_epsilon:
            # Center of gravity
            cx = m["m10"]/m["m00"]
            cy = m["m01"]/m["m00"]
        else:
            cx = 0
            cy = 0

        # central second order moments
        m["mu20"] = m["m20"] - m["m10"]*cx
        m["mu11"] = m["m11"] - m["m10"]*cy
        m["mu02"] = m["m02"] - m["m01"]*cy

        m["mu30"] = m["m30"] - cx*(3*m["mu20"] + cx*m["m10"])
        m["mu21"] = m["m21"] - cx*(2*m["mu11"] + cx*m["m01"]) - cy*m["mu20"]
        m["mu12"] = m["m12"] - cy*(2*m["mu11"] + cy*m["m10"]) - cx*m["mu02"]
        m["mu03"] = m["m03"] - cy*(3*m["mu02"] + cy*m["m01"])
        return m
    else:
        return None


def get_inert_ratio_cvx(cont):
    """Compute the inertia ratio of the convex hull of a contour

    The inertia ratio is computed from the central second order of moments
    along x (mu20) and y (mu02) via `sqrt(mu20/mu02)`.

    Parameters
    ----------
    cont: ndarray or list of ndarrays of shape (N,2)
        A 2D array that holds the contour of an event (in pixels)
        e.g. obtained using `mm.contour` where  `mm` is an instance
        of `RTDCBase`. The first and second columns of `cont`
        correspond to the x- and y-coordinates of the contour.

    Returns
    -------
    inert_ratio_cvx: float or ndarray of size N
        The inertia ratio of the contour's convex hull

    Notes
    -----
    The contour moments mu20 and mu02 are computed the same way they
    are computed in OpenCV's `moments.cpp`.


    See Also
    --------
    get_inert_ratio_raw: Compute inertia ratio of a raw contour


    References
    ----------
    - `<https://en.wikipedia.org/wiki/Image_moment#Central_moments>`__
    - `<https://github.com/opencv/opencv/blob/
      f81370232a651bdac5042efe907bcaa50a66c487/modules/imgproc/src/
      moments.cpp#L93>`__
    """
    if isinstance(cont, np.ndarray):
        # If cont is an array, it is not a list of contours,
        # because contours can have different lengths.
        cont = [cont]
        ret_list = False
    else:
        ret_list = True

    length = len(cont)

    inert_ratio_cvx = np.zeros(length, dtype=float) * np.nan

    for ii in range(length):
        try:
            chull = ssp.ConvexHull(cont[ii])
        except ssp.qhull.QhullError:
            pass
        else:
            hull = cont[ii][chull.vertices, :]
            inert_ratio_cvx[ii] = get_inert_ratio_raw(hull)

    if not ret_list:
        inert_ratio_cvx = inert_ratio_cvx[0]

    return inert_ratio_cvx


def get_inert_ratio_prnc(cont):
    """Compute principal inertia ratio of a contour

    The principal inertia ratio is rotation-invariant, which
    makes it applicable to reservoir measurements where e.g.
    cells are not aligned with the channel.

    Parameters
    ----------
    cont: ndarray or list of ndarrays of shape (N,2)
        A 2D array that holds the contour of an event (in pixels)
        e.g. obtained using `mm.contour` where  `mm` is an instance
        of `RTDCBase`. The first and second columns of `cont`
        correspond to the x- and y-coordinates of the contour.

    Returns
    -------
    inert_ratio_prnc: float or ndarray of size N
        The principal inertia ratio of the contour
    """
    if isinstance(cont, np.ndarray):
        # If cont is an array, it is not a list of contours,
        # because contours can have different lengths.
        cont = [cont]
        ret_list = False
    else:
        ret_list = True

    length = len(cont)

    inert_ratio_prnc = np.zeros(length, dtype=float) * np.nan

    for ii in range(length):
        moments = cont_moments_cv(cont[ii])
        if moments is not None:
            # orientation of the contour
            orient = 0.5 * np.arctan2(2 * moments['mu11'],
                                      moments['mu02'] - moments['mu20'])
            # require floating point array (only copy if necessary)
            cc = np.array(cont[ii], dtype=float, copy=False)
            # rotate contour
            rho = np.sqrt(cc[:, 0]**2 + cc[:, 1]**2)
            phi = np.arctan2(cc[:, 1], cc[:, 0]) + orient + np.pi / 2
            cc[:, 0] = rho * np.cos(phi)
            cc[:, 1] = rho * np.sin(phi)
            # compute inertia ratio of rotated contour
            mprnc = cont_moments_cv(cc)
            inert_ratio_prnc[ii] = np.sqrt(mprnc["mu20"] / mprnc["mu02"])

    if not ret_list:
        inert_ratio_prnc = inert_ratio_prnc[0]

    return inert_ratio_prnc


def get_inert_ratio_raw(cont):
    """Compute the inertia ratio of a contour

    The inertia ratio is computed from the central second order of moments
    along x (mu20) and y (mu02) via `sqrt(mu20/mu02)`.

    Parameters
    ----------
    cont: ndarray or list of ndarrays of shape (N,2)
        A 2D array that holds the contour of an event (in pixels)
        e.g. obtained using `mm.contour` where  `mm` is an instance
        of `RTDCBase`. The first and second columns of `cont`
        correspond to the x- and y-coordinates of the contour.

    Returns
    -------
    inert_ratio_raw: float or ndarray of size N
        The inertia ratio of the contour

    Notes
    -----
    The contour moments mu20 and mu02 are computed the same way they
    are computed in OpenCV's `moments.cpp`.

    See Also
    --------
    get_inert_ratio_cvx: Compute inertia ratio of the convex hull of
                         a contour

    References
    ----------
    - `<https://en.wikipedia.org/wiki/Image_moment#Central_moments>`__
    - `<https://github.com/opencv/opencv/blob/
      f81370232a651bdac5042efe907bcaa50a66c487/modules/imgproc/src/
      moments.cpp#L93>`__
    """
    if isinstance(cont, np.ndarray):
        # If cont is an array, it is not a list of contours,
        # because contours can have different lengths.
        cont = [cont]
        ret_list = False
    else:
        ret_list = True

    length = len(cont)

    inert_ratio_raw = np.zeros(length, dtype=float) * np.nan

    for ii in range(length):
        moments = cont_moments_cv(cont[ii])
        if moments is not None:
            inert_ratio_raw[ii] = np.sqrt(moments["mu20"]/moments["mu02"])

    if not ret_list:
        inert_ratio_raw = inert_ratio_raw[0]

    return inert_ratio_raw


def get_tilt(cont):
    """Compute tilt of raw contour relative to channel axis

    Parameters
    ----------
    cont: ndarray or list of ndarrays of shape (N,2)
        A 2D array that holds the contour of an event (in pixels)
        e.g. obtained using `mm.contour` where  `mm` is an instance
        of `RTDCBase`. The first and second columns of `cont`
        correspond to the x- and y-coordinates of the contour.

    Returns
    -------
    tilt: float or ndarray of size N
        Tilt of the contour in the interval [0, PI/2]

    References
    ----------
    - `<https://en.wikipedia.org/wiki/Image_moment#Examples_2>`__
    """
    if isinstance(cont, np.ndarray):
        # If cont is an array, it is not a list of contours,
        # because contours can have different lengths.
        cont = [cont]
        ret_list = False
    else:
        ret_list = True

    length = len(cont)

    tilt = np.zeros(length, dtype=float) * np.nan

    for ii in range(length):
        moments = cont_moments_cv(cont[ii])
        if moments is not None:
            # orientation of the contour
            oii = 0.5 * np.arctan2(2 * moments['mu11'],
                                   moments['mu02'] - moments['mu20'])
            # +PI/2 because relative to channel axis
            tilt[ii] = oii + np.pi/2

    # restrict to interval [0,PI/2]
    tilt = np.mod(tilt, np.pi)
    # avoid nans
    valid = ~np.isnan(tilt)
    wrap = tilt[valid]
    wrap[wrap > np.pi/2] -= np.pi
    tilt[valid] = wrap
    tilt = np.abs(tilt)

    if not ret_list:
        tilt = tilt[0]

    return tilt
