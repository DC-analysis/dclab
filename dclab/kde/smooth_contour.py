from __future__ import annotations

import threading

import numpy as np

from .base import KernelDensityEstimator
from .binning import bin_width_percentile


def find_smooth_contour_spacing(
        ds_list,
        xax: str,
        yax: str,
        xrange: list[float] | tuple[float, float],
        yrange: list[float] | tuple[float, float],
        quantiles: list[float] | tuple[float],
        xscale: str = "linear",
        yscale: str = "linear",
        kde_type: str = "histogram",
        kde_kwargs: dict | None = None,
        max_iter: int = 15,
        abort_event: threading.Event | None = None,
) -> dict:
    """Determine contour spacing values for visually pleasing contours

    The algorithm reduces the "kinks" in contours.

    Parameters
    ----------
    ds_list:
        list of :class:`.RTDCBase` instances for which smooth contours
        should be found
    xax, yax:
        X- and Y-axis of the contour plot
    xrange, yrange:
        Plotting range of the contour plot
    quantiles:
        Data quantiles for which contour lines should be computed; only
        the highest quantile is used for smoothing
    xscale, yscale:
        "linear" or "log" scale of the contour plot axes
    kde_type:
        Kernel density estimate method to use
    kde_kwargs:
        Custom arguments for the kernel density estimate method
    max_iter:
        Maximum number of iterations to perform before returning
    abort_event:
        Optional event for prematurely stopping the iteration

    Returns
    -------
    result:
        Distionary containing the iteration result:

            - *total iterations*: iterations performed
            - *success*: whether the smoothing succeeded
            - *reason*: reason for success or failure
            - *corners found*: whether a contour touched the plot boundary
            - *spacing x*: contour spacing along x
            - *spacing y*: contour spacing along y

    TODO
    ----
    - Exclude datasets in subsequent iterations when conditions already met
    - Include all quantiles (and exclude when conditions met)
    - Turn smoothing angle of 23° into keyword argument
    - Turn maximum contour length into keyword argument (and possibly make
      inversely dependent on quantile value)
    """
    kde_list = [KernelDensityEstimator(ds) for ds in ds_list]
    kde_kwargs = kde_kwargs or {}

    # Make an initial estimate
    spacings = []
    for ax, scale, rang in [(xax, xscale, xrange), (yax, yscale, yrange)]:
        spi = []
        for kdi, ds in zip(kde_list, ds_list):
            data = ds[ax]
            use = ds.filter.all & (rang[0] <= data) & (data <= rang[1])
            a = data[use]
            if a.size:
                spi.append(kdi.estimate_spacing(
                    a=a,
                    feat=ax,
                    scale=scale,
                    method=bin_width_percentile,
                ))
        if spi:
            spacings.append(np.min(spi))
        else:
            spacings.append(1)
    xacc, yacc = spacings

    results = {
        "total iterations": 0,
        "success": False,
    }

    corners_found = False
    for _ in range(max_iter):  # hard-limit is 15 iterations
        corners_found = False

        if abort_event is not None and abort_event.is_set():
            results["reason"] = "abort event"
            break

        results["total iterations"] += 1

        # maximum difference of opening angle from 180° [rad]
        max_dphi = 0
        max_length = np.inf
        for kdi, ds in zip(kde_list, ds_list):
            if abort_event is not None and abort_event.is_set():
                break
            # Compute the contour for the highest percentile of the plot.
            cc = kdi.get_contour_lines(
                quantiles=[[np.max(quantiles)]],
                xax=xax,
                yax=yax,
                xacc=xacc,
                yacc=yacc,
                xscale=xscale,
                yscale=yscale,
                kde_type=kde_type,
                kde_kwargs=kde_kwargs)[0][0]
            angles = compute_contour_opening_angles(
                contour=cc,
                xrange=xrange,
                yrange=yrange,
                xscale=xscale,
                yscale=yscale)

            if (np.allclose(np.abs(angles[0]), np.pi / 2, atol=0.001, rtol=0)
                    and np.all(angles[1:6] == 0)):
                # We have probably encountered a contour at the boundary
                # of the plot. This is ok.
                corners = np.abs((np.abs(angles) - np.pi / 2)) < 0.001
                dphi = np.max(np.abs(angles[~corners]))
                corners_found = True
            else:
                dphi = np.max(np.abs(angles))

            max_dphi = max(max_dphi, dphi)
            max_length = min(max_length, len(cc))
        else:
            results["reason"] = "maximum iterations reached"
            results["success"] = False

        if max_dphi < np.deg2rad(23):
            # A contour is considered smooth when the maximum angle between
            # adjacent line segments is smaller than 23°.
            results["reason"] = "target opening angle reached"
            results["success"] = True
            break
        elif max_length > 100:
            results["reason"] = "maximum contour length reached"
            results["success"] = True
            break
        else:
            xacc /= 2
            yacc /= 2

    results["corners found"] = corners_found
    results["spacing x"] = xacc
    results["spacing y"] = yacc
    return results


def compute_contour_opening_angles(contour, xrange, yrange, xscale, yscale):
    """For each point of the contour, compute the opening angle

    This takes the visible plot area into account.
    """
    cc = np.array(contour, copy=True)
    if not np.all(cc[0] == cc[-1]):
        cc = np.resize(cc, (len(contour)+1, 2))
    # Normalize contour
    cc[:, 0] = (cc[:, 0] - xrange[0]) / (xrange[1] - xrange[0])
    cc[:, 1] = (cc[:, 1] - yrange[0]) / (yrange[1] - yrange[0])
    # apply scale
    assert xscale in ["log", "linear"]
    if xscale == "log":
        cc[:, 0] = np.log10(cc[:, 0])
    assert yscale in ["log", "linear"]
    if yscale == "log":
        cc[:, 1] = np.log10(cc[:, 1])
    opang = np.zeros(len(cc)-1, dtype=float)
    for jj, c0 in enumerate(cc[:-1]):  # we have a closed contour
        cl = cc[:-1][jj - 1]
        cr = cc[jj + 1]
        # vector a
        a = np.array(cl) - np.array(c0)
        # vector b
        b = np.array(cr) - np.array(c0)
        absa = np.sqrt(np.sum(a ** 2))
        absb = np.sqrt(np.sum(b ** 2))
        denom = absa * absb
        # avoid division by zero warnings (denom == 0)
        phi = np.arccos(np.sum(a * b) / (denom or np.nan))
        if np.abs(phi) > np.pi/2:
            phi -= np.sign(phi) * np.pi
        opang[jj] = phi
    return opang
