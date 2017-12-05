#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from dclab.features import emodulus


def test_simple_emod():
    x = np.linspace(0, 250, 100)
    y = np.linspace(0, 0.1, 100)
    x, y = np.meshgrid(x, y)

    emod = emodulus.get_emodulus(area=x,
                                 deformation=y,
                                 medium="CellCarrier",
                                 channel_width=30,
                                 flow_rate=0.16,
                                 px_um=0,  # withour pixelation correction
                                 temperature=23)

    assert np.allclose(emod[10, 50], 0.93276932212481323)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
