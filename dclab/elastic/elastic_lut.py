#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Computation of apparent Young's modulus for RT-DC measurements"""
from __future__ import division, print_function, unicode_literals

import numpy as np
import pkg_resources
import scipy.interpolate as spint

from .viscosity import get_viscosity


def get_elasticity(area, deformation, medium="CellCarrier",
                   channel_width=20.0, flow_rate=0.16, temperature=23.0):
    """Compute apparent Young's modulus
    
    Parameters
    ----------
    area: float or ndarray
        Apparent (2D image) area  in µm of the event
    deformation: float or ndarray 
        The deformation (1-circularity) of the event
    medium: str or float
        The medium to compute the viscosity for. If a string
        in ["CellCarrier", "CellCarrierB"] is given, the viscosity
        will be computed. If a float is given, this value will be
        used as the viscosity. 
    channel_width: float
        The channel width in µm
    flow_rate: float
        Flow rate in µl/s
    temperature: float or ndarray
        Temperature in °C

    Returns
    -------
    elasticity: float or ndarray
        Apparent elasticity
    
    Notes
    -----
    The computation of the elasticity takes into account corrections for
    the viscosity (medium, channel width, flow rate, and temperature) and
    corrections for pixelation of the area and the deformation which are
    computed from a (pixelated) image.
    
    See Also
    --------
    dclab.elast.viscosity.get_viscosity
    """
    # Get lut data
    lut_path = pkg_resources.resource_filename("dclab.elastic", "elastic_lut.txt")
    lut = np.loadtxt(lut_path)
    # These meta data are the ismulation parameters of the lut 
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

    # Normalize interpolation data such that the spacing for
    # area and deformation is about the same during interpolation.
    area_norm = lut[:,0].max()
    lut[:,0] = normalize(lut[:,0], area_norm)
    area = normalize(area, area_norm)
    
    defo_norm = lut[:,1].max()
    lut[:,1] = normalize(lut[:,1], defo_norm)
    deformation = normalize(deformation, defo_norm)
    
    # Perform interpolation
    emod = spint.griddata((lut[:,0], lut[:,1]), lut[:,2],
                          (area, deformation),
                          method='linear')
    return emod


def normalize(data, dmax):
    return data / dmax

