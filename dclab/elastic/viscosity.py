#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Viscosity computation for RT-DC media"""
from __future__ import division, print_function, unicode_literals


def get_viscosity(medium="CellCarrier", channel_width=20.0, flow_rate=0.16,
                  temperature=23.0):
    """Returns the viscosity for RT-DC-specific media

    
    Parameters
    ----------
    medium: str
        The medium to compute the viscosity for.
        One of ["CellCarrier", "CellCarrier B"].
    channel_width: float
        The channel width in µm
    flow_rate: float
        Flow rate in µl/s
    temperature: float or ndarray
        Temperature in °C

    Returns
    -------
    viscosity: float or ndarray
        Viscosity in mPa*s

    Notes
    -----
    The CellCarrier A and B media are optimized for RT-DC measurements.
    """
    assert medium.lower() in ["cellcarrier", "cellcarrier b"]
    
    # convert flow_rate from µl/s to m³/s
    # convert channel_width from µm to m
    term1 = 1.1856*6*flow_rate*1e-9/(channel_width*1e-6)**3 * 2/3
    
    if medium == "CellCarrier":
        temp_corr = (temperature/23.2)**-.866
        term2 = 0.6771/0.5928+0.2121/(0.5928*0.677)
        eta = 0.179*(term1*term2)**(0.677-1)*temp_corr*1e3
    elif medium == "CellCarrier B":
        temp_corr = (temperature/23.6)**-.866
        term2 = 0.6771/0.5928+0.2121/(0.5928*0.634)
        eta = 0.360*(term1*term2)**(0.634-1)*temp_corr*1e3

    return eta

