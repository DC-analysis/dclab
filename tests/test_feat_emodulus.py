#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from dclab.features import emodulus


def test_simple_emod():
    x = np.linspace(0, 250, 100)
    y = np.linspace(0, 0.1, 100)
    x, y = np.meshgrid(x, y)

    emod = emodulus.get_emodulus(area_um=x,
                                 deform=y,
                                 medium="CellCarrier",
                                 channel_width=30,
                                 flow_rate=0.16,
                                 px_um=0,  # without pixelation correction
                                 temperature=23)

    assert np.allclose(emod[10, 50], 1.1875799054283109)
    assert np.allclose(emod[50, 50], 0.55281291845478731)
    assert np.allclose(emod[80, 50], 0.45678187644969814)

    assert np.allclose(emod[10, 80], 1.5744560306483262)
    assert np.allclose(emod[50, 80], 0.73534561544655519)
    assert np.allclose(emod[80, 80], 0.60737083178222251)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
