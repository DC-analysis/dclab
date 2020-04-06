#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import pytest

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


@pytest.mark.filterwarnings(
    'ignore::dclab.features.emodulus.KnowWhatYouAreDoingWarning')
def test_extrapolate():
    """Test whether spline interpolation gives reasonable results"""
    lut, _ = emodulus.load_lut("emodulus_lut.txt")

    area_norm = lut[:, 0].max()
    emodulus.normalize(lut[:, 0], area_norm)

    deform_norm = lut[:, 1].max()
    emodulus.normalize(lut[:, 1], deform_norm)

    np.random.seed(47)
    more_than_5perc = []
    valid_ones = 0

    for _ in range(100):
        # pick a few values from the LUT
        ids = np.random.randint(0, lut.shape[0], 10)
        area_um = lut[ids, 0]
        deform = lut[ids, 1]
        # set the emodulus to zero
        emod = np.nan * np.zeros(deform.size)
        # "extrapolate" within the grid using the spline
        emodulus.extrapolate_emodulus(
            lut=lut,
            area_um=area_um,
            deform=deform,
            emod=emod,
            deform_norm=deform_norm,
            inplace=True)
        valid = ~np.isnan(emod)
        valid_ones += np.sum(valid)
        res = np.abs(lut[ids, 2] - emod)[valid]/lut[ids, 2][valid]
        if np.sum(res > .05):
            more_than_5perc.append([ids, res])

    assert len(more_than_5perc) == 0
    assert valid_ones == 151


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
