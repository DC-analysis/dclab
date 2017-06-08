#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Computation of apparent Young's modulus for RT-DC measurements"""
from __future__ import division, print_function, unicode_literals

import numpy as np
import pkg_resources
import scipy.interpolate as spint

from .viscosity import get_viscosity



def get_elasticity(area, deformation, medium="CellCarrier",
                   channel_width=20.0, flow_rate=0.16, px_um=0.34,
                   temperature=23.0, copy=True):
    """Compute apparent Young's modulus
    
    Parameters
    ----------
    area: float or ndarray
        Apparent (2D image) area in µm² of the event(s)
    deformation: float or ndarray 
        The deformation (1-circularity) of the event(s)
    medium: str or float
        The medium to compute the viscosity for. If a string
        in ["CellCarrier", "CellCarrier B"] is given, the viscosity
        will be computed. If a float is given, this value will be
        used as the viscosity. 
    channel_width: float
        The channel width in µm
    flow_rate: float
        Flow rate in µl/s
    px_um: float
        The detector pixel size in µm. Set this value to zero
        to disable pixelation correction.
    temperature: float or ndarray
        Temperature in °C of the event(s)
    copy: bool
        Copy input arrays. If set to false, input arrays are
        overridden.

    Returns
    -------
    elasticity: float or ndarray
        Apparent Young's modulus in kPa
    
    Notes
    -----
    The computation of the elasticity takes into account corrections for
    the viscosity (medium, channel width, flow rate, and temperature) and
    corrections for pixelation of the area and the deformation which are
    computed from a (pixelated) image.
    
    See Also
    --------
    dclab.elast.viscosity.get_viscosity: compute viscosity for known media
    pixcorr_deformation: perform pixelation correction with triple-exponential
    """
    # copy input arrays so we can use in-place calculations
    deformation = np.array(deformation, copy=copy, dtype=float)
    area = np.array(area, copy=copy, dtype=float)
    # Get lut data
    lut_path = pkg_resources.resource_filename("dclab.elastic", "elastic_lut.txt")
    lut = np.loadtxt(lut_path)
    # These meta data are the simulation parameters of the lut 
    lut_channel_width = 20.0
    lut_flow_rate = 0.04
    lut_visco = 15.0
    ## Corrections
    # We correct the lut, because it contains less points than
    # the event data to implement. Furthermore, the lut could
    # be cached in the future, if this takes up a lot of time.
    if lut_channel_width != channel_width:
        # convert lut area axis to match channel width
        lut[:,0] *= (channel_width/lut_channel_width)**2
    # Compute viscosity
    if isinstance(medium, (float, int)):
        visco = medium
    else:
        visco = get_viscosity(medium=medium, channel_width=channel_width,
                              flow_rate=flow_rate, temperature=temperature)
    # Correct elastic modulus    
    lut[:,2] *= (flow_rate/lut_flow_rate)*\
                (visco/lut_visco)*\
                (lut_channel_width/channel_width)**3 

    if px_um:
        # Correct deformation for pixelation effect (inplace).
        pixcorr_deformation(area=area, deformation=deformation,
                            px_um=px_um, inplace=True)

    # Normalize interpolation data such that the spacing for
    # area and deformation is about the same during interpolation.
    area_norm = lut[:,0].max()
    normalize(lut[:,0], area_norm)
    normalize(area, area_norm)
    
    defo_norm = lut[:,1].max()
    normalize(lut[:,1], defo_norm)
    normalize(deformation, defo_norm)
    
    # Perform interpolation
    emod = spint.griddata((lut[:,0], lut[:,1]), lut[:,2],
                          (area, deformation),
                          method='linear')
    return emod


def normalize(data, dmax):
    """Perform normalization inplace"""
    data /= dmax
    return data


def pixcorr_deformation(area, deformation, px_um=0.34, inplace=False):
    """Correct deformation for pixelation effects
    
    The contour in RT-DC measurements is computed on a
    pixelated grid. Due to sampling problems, the measured
    deformation is overestimated and must be corrected.

    Parameters
    ----------
    area: float or ndarray
        Apparent (2D image) area in µm² of the event(s)
    deformation: float or ndarray 
        The deformation (1-circularity) of the event(s)
    px_um: float
        The detector pixel size in µm.
    inplace: bool
        Change the deformation values in-place
    
    
    Returns
    -------
    deformation_corr: float or ndarray
        The corrected deformation of the event(s)
    """
    msg = "Pixelelation correction for {}um per px unavailable!".format(px_um)
    assert px_um in [0.34], msg
    
    if px_um==0.34:
        # A triple-exponential decay can be used to correct for pixelation
        # for apparent cell areas between 10 and 1250µm².
        # For 99 different radii between 0.4 μm and 20 μm circular objects were
        # simulated on a pixel grid with the pixel resolution of 340 nm/pix. At
        # each radius 1000 random starting points were created and the
        # obtained contours were analyzed in the same fashion as RT-DC data.
        # A convex hull on the contour was used to calculate the size (as area)
        # and the deformation.
        offs = 0.0012
        exp1 = 0.020*np.exp(-area/7.1)
        exp2 = 0.010*np.exp(-area/38.6)
        exp3 = 0.005*np.exp(-area/296)
        delta = offs + exp1 + exp2 + exp3
        
    if inplace:
        deformation -= delta
    else:
        deformation = deformation - delta
    
    return deformation
