#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from dclab.features import emodulus_viscosity


def test_cell_carrier():
    """Test using table from Christophs script"""
    ch_sizes = [15, 15, 15, 20, 20, 20, 20, 20, 30, 30, 30, 40, 40, 40]
    fl_rates = [0.016, 0.032, 0.048, 0.02, 0.04, 0.06,
                0.08, 0.12, 0.16, 0.24, 0.32, 0.32, 0.40, 0.60]
    temps = [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24]
    eta_a = [5.8, 4.6, 4.1, 7.1, 5.7, 5.0,
             4.5, 4.0, 5.4, 4.7, 4.3, 5.7, 5.3, 4.6]
    eta_b = [7.5, 5.8, 5.0, 9.4, 7.3, 6.3,
             5.7, 4.9, 6.9, 5.9, 5.3, 7.3, 6.7, 5.8]

    for c, f, t, a in zip(ch_sizes, fl_rates, temps, eta_a):
        eta = emodulus_viscosity.get_viscosity(medium="CellCarrier",
                                               channel_width=c,
                                               flow_rate=f,
                                               temperature=t)
        assert np.allclose(np.round(eta, 1), a)

    for c, f, t, b in zip(ch_sizes, fl_rates, temps, eta_b):
        eta = emodulus_viscosity.get_viscosity(medium="CellCarrier B",
                                               channel_width=c,
                                               flow_rate=f,
                                               temperature=t)
        assert np.allclose(np.round(eta, 1), b)


def test_water():
    """Test with data from Kestin et al, J. Phys. Chem. 7(3) 1978"""
    ref = np.array([[0, 1791.5],
                    [5, 1519.3],
                    [10, 1307.0],
                    [15, 1138.3],
                    [20, 1002.0],
                    [25, 890.2],
                    [30, 797.3],
                    [35, 719.1],
                    [40, 652.7],
                    ])
    ref[:, 1] *= 1e-3  # uPas to mPas

    res = emodulus_viscosity.get_viscosity(medium="water",
                                           temperature=ref[:, 0])
    assert np.allclose(res, ref[:, 1], rtol=8e-5, atol=0)


def test_water_range():
    # test values
    try:
        emodulus_viscosity.get_viscosity(medium="water", temperature=-1)
    except ValueError:
        pass
    else:
        assert False, "temperature should be outside valid range"

    try:
        emodulus_viscosity.get_viscosity(medium="water", temperature=41)
    except ValueError:
        pass
    else:
        assert False, "temperature should be outside valid range"
    # also test arrays
    try:
        emodulus_viscosity.get_viscosity(medium="water",
                                         temperature=np.arange(-2, 10))
    except ValueError:
        pass
    else:
        assert False, "temperature should be outside valid range"


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
