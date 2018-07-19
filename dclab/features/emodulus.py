#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Computation of apparent Young's modulus for RT-DC measurements"""
from __future__ import division, print_function, unicode_literals

import pathlib
from pkg_resources import resource_filename

import numpy as np
import scipy.interpolate as spint

from .emodulus_viscosity import get_viscosity


def convert(area_um, deform, emodulus,
            channel_width_in, channel_width_out,
            flow_rate_in, flow_rate_out,
            viscosity_in, viscosity_out,
            inplace=False):
    """convert area-deformation-emodulus triplet

    The conversion formula is described in :cite:`Mietke2015`.

    Parameters
    ----------
    area_um: ndarray
        Convex cell area [µm²]
    deform: ndarray
        Deformation
    emodulus: ndarray
        Young's Modulus [kPa]
    channel_width_in: float
        Original channel width [µm]
    channel_width_out: float
        Target channel width [µm]
    flow_rate_in: float
        Original flow rate [µl/s]
    flow_rate_in: float
        Target flow rate [µl/s]
    viscosity_in: float
        Original viscosity [mPa*s]
    viscosity_out: float
        Target viscosity [mPa*s]
    inplace: bool
        If True, override input arrays with corrected data

    Returns
    -------
    area_um_corr: ndarray
        Corrected cell area [µm²]
    deform_corr: ndarray
        Deformation (a copy if `inplace` is False)
    emodulus_corr: ndarray
        Corrected emodulus [kPa]
    """
    copy = not inplace
    # make sure area_um_corr is not an integer array
    area_um_corr = np.array(area_um, dtype=float, copy=copy)
    deform_corr = np.array(deform, copy=copy)
    emodulus_corr = np.array(emodulus, copy=copy)

    if channel_width_in != channel_width_out:
        area_um_corr *= (channel_width_out / channel_width_in)**2

    if (flow_rate_in != flow_rate_out or
            viscosity_in != viscosity_out or
            channel_width_in != channel_width_out):
        emodulus_corr *= (flow_rate_out / flow_rate_in) \
            * (viscosity_out / viscosity_in) \
            * (channel_width_in / channel_width_out)**3

    return area_um_corr, deform_corr, emodulus_corr


def corrpix_deform_delta(area_um, px_um=0.34):
    """Deformation correction term for pixelation effects

    The contour in RT-DC measurements is computed on a
    pixelated grid. Due to sampling problems, the measured
    deformation is overestimated and must be corrected.

    The correction formula is described in :cite:`Herold2017`.

    Parameters
    ----------
    area_um: float or ndarray
        Apparent (2D image) area in µm² of the event(s)
    px_um: float
        The detector pixel size in µm.
    inplace: bool
        Change the deformation values in-place

    Returns
    -------
    deform_delta: float or ndarray
        Error of the deformation of the event(s) that must be
        subtracted from `deform`.
        deform_corr = deform -  deform_delta
    """
    # A triple-exponential decay can be used to correct for pixelation
    # for apparent cell areas between 10 and 1250µm².
    # For 99 different radii between 0.4 μm and 20 μm circular objects were
    # simulated on a pixel grid with the pixel resolution of 340 nm/pix. At
    # each radius 1000 random starting points were created and the
    # obtained contours were analyzed in the same fashion as RT-DC data.
    # A convex hull on the contour was used to calculate the size (as area)
    # and the deformation.
    # The pixel size correction `pxcorr` takes into account the pixel size
    # in the pixelation correction formula.
    pxcorr = (.34 / px_um)**2
    offs = 0.0012
    exp1 = 0.020 * np.exp(-area_um * pxcorr / 7.1)
    exp2 = 0.010 * np.exp(-area_um * pxcorr / 38.6)
    exp3 = 0.005 * np.exp(-area_um * pxcorr / 296)
    delta = offs + exp1 + exp2 + exp3

    return delta


def get_emodulus(area_um, deform, medium="CellCarrier",
                 channel_width=20.0, flow_rate=0.16, px_um=0.34,
                 temperature=23.0, copy=True):
    """Compute apparent Young's modulus using a look-up table

    Parameters
    ----------
    area_um: float or ndarray
        Apparent (2D image) area [µm²] of the event(s)
    deform: float or ndarray
        The deformation (1-circularity) of the event(s)
    medium: str or float
        The medium to compute the viscosity for. If a string
        in ["CellCarrier", "CellCarrier B"] is given, the viscosity
        will be computed. If a float is given, this value will be
        used as the viscosity in mPa*s.
    channel_width: float
        The channel width [µm]
    flow_rate: float
        Flow rate [µl/s]
    px_um: float
        The detector pixel size [µm] used for pixelation correction.
        Set to zero to disable.
    temperature: float or ndarray
        Temperature [°C] of the event(s)
    copy: bool
        Copy input arrays. If set to false, input arrays are
        overridden.

    Returns
    -------
    elasticity: float or ndarray
        Apparent Young's modulus in kPa

    Notes
    -----
    - The look-up table used was computed with finite elements methods
      according to :cite:`Mokbel2017`.
    - The computation of the Young's modulus takes into account
      corrections for the viscosity (medium, channel width, flow rate,
      and temperature) :cite:`Mietke2015` and corrections for
      pixelation of the area and the deformation which are computed
      from a (pixelated) image :cite:`Herold2017`.

    See Also
    --------
    dclab.features.emodulus_viscosity.get_viscosity: compute viscosity
        for known media
    """
    # copy input arrays so we can use in-place calculations
    deform = np.array(deform, copy=copy, dtype=float)
    area_um = np.array(area_um, copy=copy, dtype=float)
    # Get lut data
    lut_path = resource_filename("dclab.features", "emodulus_lut.txt")
    with pathlib.Path(lut_path).open("rb") as lufd:
        lut = np.loadtxt(lufd)
    # These meta data are the simulation parameters of the lut
    lut_channel_width = 20.0
    lut_flow_rate = 0.04
    lut_visco = 15.0
    # Compute viscosity
    if isinstance(medium, (float, int)):
        visco = medium
    else:
        visco = get_viscosity(medium=medium, channel_width=channel_width,
                              flow_rate=flow_rate, temperature=temperature)
    # Corrections
    # We correct the lut, because it contains less points than
    # the event data. Furthermore, the lut could be cached
    # in the future, if this takes up a lot of time.
    convert(area_um=lut[:, 0],
            deform=lut[:, 1],
            emodulus=lut[:, 2],
            channel_width_in=lut_channel_width,
            channel_width_out=channel_width,
            flow_rate_in=lut_flow_rate,
            flow_rate_out=flow_rate,
            viscosity_in=lut_visco,
            viscosity_out=visco,
            inplace=True)

    if px_um:
        # Correct deformation for pixelation effect (subtract ddelt).
        ddelt = corrpix_deform_delta(area_um=area_um, px_um=px_um)
        deform -= ddelt

    # Normalize interpolation data such that the spacing for
    # area and deformation is about the same during interpolation.
    area_norm = lut[:, 0].max()
    normalize(lut[:, 0], area_norm)
    normalize(area_um, area_norm)

    defo_norm = lut[:, 1].max()
    normalize(lut[:, 1], defo_norm)
    normalize(deform, defo_norm)

    # Perform interpolation
    emod = spint.griddata((lut[:, 0], lut[:, 1]), lut[:, 2],
                          (area_um, deform),
                          method='linear')
    return emod


def normalize(data, dmax):
    """Perform normalization inplace"""
    data /= dmax
    return data
