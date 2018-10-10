#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Viscosity computation for various media"""
from __future__ import division, print_function, unicode_literals

import numpy as np


def get_viscosity(medium="CellCarrier", channel_width=20.0, flow_rate=0.16,
                  temperature=23.0):
    """Returns the viscosity for RT-DC-specific media

    Parameters
    ----------
    medium: str
        The medium to compute the viscosity for.
        One of ["CellCarrier", "CellCarrier B", "water"].
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
    - CellCarrier and CellCarrier B media are optimized for
      RT-DC measurements.
    - Values for the viscosity of water are computed using
      equation (15) from :cite:`Kestin_1978`.
    """
    if medium.lower() not in ["cellcarrier", "cellcarrier b", "water"]:
        raise ValueError("Invalid medium: {}".format(medium))

    # convert flow_rate from µl/s to m³/s
    # convert channel_width from µm to m
    term1 = 1.1856 * 6 * flow_rate * 1e-9 / (channel_width * 1e-6)**3 * 2 / 3

    if medium == "CellCarrier":
        temp_corr = (temperature / 23.2)**-0.866
        term2 = 0.6771 / 0.5928 + 0.2121 / (0.5928 * 0.677)
        eta = 0.179 * (term1 * term2)**(0.677 - 1) * temp_corr * 1e3
    elif medium == "CellCarrier B":
        temp_corr = (temperature / 23.6)**-0.866
        term2 = 0.6771 / 0.5928 + 0.2121 / (0.5928 * 0.634)
        eta = 0.360 * (term1 * term2)**(0.634 - 1) * temp_corr * 1e3
    elif medium == "water":
        # see equation (15) in Kestin et al, J. Phys. Chem. 7(3) 1978
        if np.min(temperature) < 0 or np.max(temperature) > 40:
            msg = "For water, the temperature must be in [0, 40] degC! " \
                  "Got min/max values of '{}'.".format(np.min(temperature),
                                                       np.max(temperature))
            raise ValueError(msg)
        eta0 = 1.002  # [mPa]
        right = (20-temperature) / (temperature + 96) \
            * (+ 1.2364
               - 1.37e-3 * (20 - temperature)
               + 5.7e-6 * (20 - temperature)**2
               )
        eta = eta0 * 10**right
    return eta
