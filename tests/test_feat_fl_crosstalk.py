#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from dclab.features.fl_crosstalk import correct_crosstalk


def test_simple_crosstalk():
    fl1 = np.array([1.1, 3.1, 6.3])
    fl2 = np.array([1.6, 30.1, 16.3])
    fl3 = np.array([10.3, 7.1, 8.9])

    ct21 = .1
    ct31 = .5
    ct12 = .03
    ct32 = .25
    ct13 = .01
    ct23 = .2

    ct11 = 1
    ct22 = 1
    ct33 = 1

    # compute cross-talked data
    fl1_bleed = ct11 * fl1 + ct21 * fl2 + ct31 * fl3
    fl2_bleed = ct12 * fl1 + ct22 * fl2 + ct32 * fl3
    fl3_bleed = ct13 * fl1 + ct23 * fl2 + ct33 * fl3

    # obtain crosstalk-corrected data
    fl1_ctc = correct_crosstalk(fl1=fl1_bleed,
                                fl2=fl2_bleed,
                                fl3=fl3_bleed,
                                fl_channel=1,
                                ct21=ct21,
                                ct31=ct31,
                                ct12=ct12,
                                ct32=ct32,
                                ct13=ct13,
                                ct23=ct23)

    fl2_ctc = correct_crosstalk(fl1=fl1_bleed,
                                fl2=fl2_bleed,
                                fl3=fl3_bleed,
                                fl_channel=2,
                                ct21=ct21,
                                ct31=ct31,
                                ct12=ct12,
                                ct32=ct32,
                                ct13=ct13,
                                ct23=ct23)

    fl3_ctc = correct_crosstalk(fl1=fl1_bleed,
                                fl2=fl2_bleed,
                                fl3=fl3_bleed,
                                fl_channel=3,
                                ct21=ct21,
                                ct31=ct31,
                                ct12=ct12,
                                ct32=ct32,
                                ct13=ct13,
                                ct23=ct23)

    assert np.allclose(fl1, fl1_ctc)
    assert np.allclose(fl2, fl2_ctc)
    assert np.allclose(fl3, fl3_ctc)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
