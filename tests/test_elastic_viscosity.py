#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import os
from os.path import abspath, dirname, join
import shutil
import sys
import tempfile
import warnings
import zipfile


# Add parent directory to beginning of path variable
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from dclab.elastic import viscosity


def test_cell_carrier():
    """Test using table from Christophs script"""
    ch_sizes = [15,15,15,20,20,20,20,20,30,30,30,40,40,40]
    fl_rates = [0.016,0.032,0.048,0.02,0.04,0.06,0.08,0.12,0.16,0.24,0.32,0.32,0.40,0.60]
    temps = [24,24,24,24,24,24,24,24,24,24,24,24,24,24]
    eta_a = [5.8,4.6,4.1,7.1,5.7,5.0,4.5,4.0,5.4,4.7,4.3,5.7,5.3,4.6]
    eta_b = [7.5,5.8,5.0,9.4,7.3,6.3,5.7,4.9,6.9,5.9,5.3,7.3,6.7,5.8]
    
    for c, f, t, a in zip(ch_sizes, fl_rates, temps, eta_a):
        eta = viscosity.get_viscosity(medium="CellCarrier",
                                       channel_width=c,
                                       flow_rate=f,
                                       temperature=t)
        assert np.allclose(np.round(eta, 1), a)

    for c, f, t, b in zip(ch_sizes, fl_rates, temps, eta_b):
        eta = viscosity.get_viscosity(medium="CellCarrier B",
                                       channel_width=c,
                                       flow_rate=f,
                                       temperature=t)
        assert np.allclose(np.round(eta, 1), b)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
