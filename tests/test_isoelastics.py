#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import pathlib

import numpy as np

from dclab import isoelastics as iso
from dclab.features import emodulus


def get_isofile(name="example_isoelastics.txt"):
    thisdir = pathlib.Path(__file__).parent
    return thisdir / "data" / name


def test_pixel_err():
    i1 = iso.Isoelastics([get_isofile()])
    isoel = i1._data["analytical"]["area_um"]["deform"]["isoelastics"]
    px_um = .10
    # add the error
    isoel_err = i1.add_px_err(isoel=isoel,
                              col1="area_um",
                              col2="deform",
                              px_um=px_um,
                              inplace=False)
    # remove the error manually
    isoel_corr = []
    for iss in isoel_err:
        iss = iss.copy()
        iss[:, 1] -= emodulus.corrpix_deform_delta(area_um=iss[:, 0],
                                                   px_um=px_um)
        isoel_corr.append(iss)

    for ii in range(len(isoel)):
        assert not np.allclose(isoel[ii], isoel_err[ii])
        assert np.allclose(isoel[ii], isoel_corr[ii])


def test_circ():
    i1 = iso.Isoelastics([get_isofile()])
    iso1 = i1._data["analytical"]["area_um"]["deform"]["isoelastics"]
    iso2 = i1._data["analytical"]["area_um"]["circ"]["isoelastics"]
    assert np.allclose(iso1[0][:, 1], 1 - iso2[0][:, 1])


def test_circ_get():
    i1 = iso.Isoelastics([get_isofile()])
    iso_circ = i1.get(col1="area_um",
                      col2="circ",
                      method="analytical",
                      channel_width=15,
                      flow_rate=0.04,
                      viscosity=15)
    iso_deform = i1.get(col1="area_um",
                        col2="deform",
                        method="analytical",
                        channel_width=15,
                        flow_rate=0.04,
                        viscosity=15)
    for ii in range(len(iso_circ)):
        isc = iso_circ[ii]
        isd = iso_deform[ii]
        assert np.allclose(isc[:, 0], isd[:, 0])
        assert np.allclose(isc[:, 1], 1 - isd[:, 1])


def test_convert():
    i1 = iso.Isoelastics([get_isofile()])
    isoel = i1._data["analytical"]["area_um"]["deform"]["isoelastics"]
    isoel15 = i1.convert(isoel=isoel,
                         col1="area_um",
                         col2="deform",
                         channel_width_in=20,
                         channel_width_out=15,
                         flow_rate_in=0.04,
                         flow_rate_out=0.04,
                         viscosity_in=15,
                         viscosity_out=15)
    # These values were taken from previous isoelasticity files
    # used in Shape-Out.
    assert np.allclose(isoel15[0][:, 2], 7.11111111e-01)
    assert np.allclose(isoel15[1][:, 2], 9.48148148e-01)
    # area_um
    assert np.allclose(isoel15[0][1, 0], 2.245995843750000276e+00)
    assert np.allclose(isoel15[0][9, 0], 9.954733499999999680e+00)
    assert np.allclose(isoel15[1][1, 0], 2.247747243750000123e+00)
    # deform
    assert np.allclose(isoel15[0][1, 1], 5.164055600000000065e-03)
    assert np.allclose(isoel15[0][9, 1], 2.311524599999999902e-02)
    assert np.allclose(isoel15[1][1, 1], 2.904264599999999922e-03)


def test_convert_error():
    i1 = iso.Isoelastics([get_isofile()])
    isoel = i1.get(col1="area_um",
                   col2="deform",
                   method="analytical",
                   channel_width=15)

    kwargs = dict(channel_width_in=15,
                  channel_width_out=20,
                  flow_rate_in=.12,
                  flow_rate_out=.08,
                  viscosity_in=15,
                  viscosity_out=15)

    try:
        i1.convert(isoel=isoel,
                   col1="deform",
                   col2="deform",
                   **kwargs)
    except ValueError:
        pass
    else:
        assert False, "identical columns"

    try:
        i1.convert(isoel=isoel,
                   col1="deform",
                   col2="circ",
                   **kwargs)
    except ValueError:
        pass
    else:
        assert False, "area_um required"

    try:
        i1.convert(isoel=isoel,
                   col1="deform",
                   col2="volume",
                   **kwargs)
    except ValueError:
        pass
    else:
        assert False, "undefined column volume"


def test_data_slicing():
    i1 = iso.Isoelastics([get_isofile()])
    iso1 = i1._data["analytical"]["area_um"]["deform"]["isoelastics"]
    iso2 = i1._data["analytical"]["deform"]["area_um"]["isoelastics"]
    for ii in range(len(iso1)):
        assert np.all(iso1[ii][:, 2] == iso2[ii][:, 2])
        assert np.all(iso1[ii][:, 0] == iso2[ii][:, 1])
        assert np.all(iso1[ii][:, 1] == iso2[ii][:, 0])


def test_data_structure():
    i1 = iso.Isoelastics([get_isofile()])
    # basic import
    assert "analytical" in i1._data
    assert "deform" in i1._data["analytical"]
    assert "area_um" in i1._data["analytical"]["deform"]
    assert "area_um" in i1._data["analytical"]
    assert "deform" in i1._data["analytical"]["area_um"]
    # circularity
    assert "circ" in i1._data["analytical"]
    assert "area_um" in i1._data["analytical"]["circ"]
    assert "area_um" in i1._data["analytical"]
    assert "circ" in i1._data["analytical"]["area_um"]
    # metadata
    meta1 = i1._data["analytical"]["area_um"]["deform"]["meta"]
    meta2 = i1._data["analytical"]["deform"]["area_um"]["meta"]
    assert meta1 == meta2


def test_get():
    i1 = iso.Isoelastics([get_isofile()])
    data = i1.get(col1="area_um",
                  col2="deform",
                  channel_width=20,
                  flow_rate=0.04,
                  viscosity=15,
                  method="analytical")
    refd = i1._data["analytical"]["area_um"]["deform"]["isoelastics"]

    for a, b in zip(data, refd):
        assert np.all(a == b)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
