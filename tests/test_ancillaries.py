#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import pathlib
import time

import numpy as np
import pytest

import dclab
from dclab.rtdc_dataset import ancillaries

from helper_methods import example_data_dict, retrieve_data, \
    example_data_sets, cleanup


def test_0basic():
    ds = dclab.new_dataset(retrieve_data(example_data_sets[1]))
    for cc in ['fl1_pos',
               'frame',
               'size_x',
               'size_y',
               'contour',
               'area_cvx',
               'circ',
               'image',
               'trace',
               'fl1_width',
               'nevents',
               'pos_x',
               'pos_y',
               'fl1_area',
               'fl1_max',
               ]:
        assert cc in ds

    # ancillaries
    for cc in ["deform",
               "area_um",
               "aspect",
               "frame",
               "index",
               "time",
               ]:
        assert cc in ds

    cleanup()


def test_0error():
    keys = ["circ"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    try:
        ds["unknown_column"]
    except KeyError:
        pass
    else:
        raise ValueError("Should have raised KeyError!")


def test_aspect():
    # Aspect ratio of the data
    ds = dclab.new_dataset(retrieve_data("rtdc_data_traces_video_bright.zip"))
    aspect = ds["aspect"]
    assert np.sum(aspect > 1) == 904
    assert np.sum(aspect < 1) == 48
    cleanup()


def test_area_ratio():
    ds = dclab.new_dataset(retrieve_data("rtdc_data_traces_video.zip"))
    comp_ratio = ds["area_ratio"]
    # The convex area is always >= the raw area
    assert np.all(comp_ratio >= 1)
    assert np.allclose(comp_ratio[0], 1.0196464)
    cleanup()


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'ancillaries.ancillary_feature.'
                            + 'BadFeatureSizeWarning')
def test_brightness():
    # Brightness of the image
    ds = dclab.new_dataset(retrieve_data("rtdc_data_traces_video_bright.zip"))
    # This is something low-level and should not be done in a script.
    # Remove the brightness columns from RTDCBase to force computation with
    # the image and contour columns.
    real_avg = ds._events.pop("bright_avg")
    real_sd = ds._events.pop("bright_sd")
    # This will cause a zero-padding warning:
    comp_avg = ds["bright_avg"]
    comp_sd = ds["bright_sd"]
    idcompare = ~np.isnan(comp_avg)
    # ignore first event (no image data)
    idcompare[0] = False
    assert np.allclose(real_avg[idcompare], comp_avg[idcompare])
    assert np.allclose(real_sd[idcompare], comp_sd[idcompare])
    cleanup()


def test_contour_basic():
    ds1 = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_mask_contour.zip"))
    # export all data except for contour data
    features = ds1.features
    features.remove("contour")
    dspath = pathlib.Path(ds1.path)
    tempout = dspath.parent / (dspath.name + "without_contour.rtdc")
    ds1.export.hdf5(tempout, features=features)
    ds2 = dclab.new_dataset(tempout)

    for ii in range(len(ds1)):
        cin = ds1["contour"][ii]
        cout = ds2["contour"][ii]
        # simple presence test
        for ci in cin:
            assert ci in cout
        # order
        for ii in range(1, len(cin)):
            c2 = np.roll(cin, ii, axis=0)
            if np.all(c2 == cout):
                break
        else:
            assert False, "contours not matching, check orientation?"
    cleanup()


def test_deform():
    keys = ["circ"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    assert np.allclose(ds["deform"], 1 - ds["circ"])


def test_emodulus():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ds.config["calculation"] = {"emodulus medium": "CellCarrier",
                                "emodulus model": "elastic sphere",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    t1 = time.time()
    assert "emodulus" in ds
    t2 = time.time()

    t3 = time.time()
    ds["emodulus"]
    t4 = time.time()
    assert t4 - t3 > t2 - t1


def test_emodulus_area():
    # computes "area_um" from "area_cvx"
    keys = ["area_cvx", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    # area can be computed from areapix
    ds.config["imaging"]["pixel size"] = .34
    assert "area_um" in ds
    assert "emodulus" not in ds, "not config for emodulus"
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["calculation"] = {"emodulus medium": "CellCarrier",
                                "emodulus model": "elastic sphere",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    assert "emodulus" in ds


def test_emodulus_none():
    keys = ["area_msd", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    assert "emodulus" not in ds, "not config for emodulus"
    ds.config["calculation"] = {"emodulus medium": "CellCarrier",
                                "emodulus model": "elastic sphere",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    assert "emodulus" not in ds, "column 'area_um' should be missing"


def test_emodulus_none2():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    assert "emodulus" not in ds, "not config for emodulus"
    ds.config["calculation"] = {"emodulus medium": "CellCarrier",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    assert "emodulus" not in ds, "emodulus model should be missing"


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.event_contour.'
                            + 'NoContourDataWarning')
def test_fl_crosstalk_2chan():
    ds = dclab.new_dataset(retrieve_data("rtdc_data_traces_2flchan.zip"))
    # simple example
    analysis = {"calculation": {"crosstalk fl12": 0,
                                "crosstalk fl21": .1}}
    ds.config.update(analysis)
    # normalization is c11 = c22 = c33 = 1
    assert np.allclose(ds["fl2_max"], ds["fl2_max_ctc"])
    assert not np.allclose(ds["fl1_max"], ds["fl1_max_ctc"])
    # advanced example
    ct12 = .5
    ct21 = .3
    analysis2 = {"calculation": {"crosstalk fl12": ct12,
                                 "crosstalk fl21": ct21}}
    # AncillaryColumn uses hashes to check whether a particular calculation
    # was already performed. Thus, just updating the config will trigger
    # a new crosstalk correction once the data is requested.
    ds.config.update(analysis2)
    fl1_max = ds["fl1_max_ctc"] + ct21 * ds["fl2_max_ctc"]
    fl2_max = ds["fl2_max_ctc"] + ct12 * ds["fl1_max_ctc"]
    assert np.allclose(fl1_max, ds["fl1_max"])
    assert np.allclose(fl2_max, ds["fl2_max"])


def test_fl_crosstalk_3chanvs2chan():
    data = {"fl1_max": np.linspace(1, 1.1, 10),
            "fl2_max": np.linspace(0, 4.1, 10),
            "fl3_max": np.linspace(3, 2.5, 10),
            }
    ds = dclab.new_dataset(data)
    analysis = {"calculation": {"crosstalk fl12": .4,
                                "crosstalk fl21": .05,
                                }}
    ds.config.update(analysis)
    assert "fl2_max_ctc" in ds
    try:
        ds["fl2_max_ctc"]
    except ancillaries.af_fl_max_ctc.MissingCrosstalkMatrixElementsError:
        pass
    else:
        assert False, "Crosstalk correction from missing data should not work"
    # add missing matrix elements
    analysis = {"calculation": {"crosstalk fl13": .1,
                                "crosstalk fl23": .7,
                                "crosstalk fl31": .2,
                                "crosstalk fl32": .2,
                                }}
    ds.config.update(analysis)
    ds["fl1_max_ctc"]
    ds["fl2_max_ctc"]
    ds["fl3_max_ctc"]
    ds.config.update(analysis)


def test_fl_crosstalk_missing():
    data = {"fl1_max": np.linspace(1, 1.1, 10),
            "fl2_max": np.linspace(0, 4.1, 10),
            }
    ds = dclab.new_dataset(data)
    analysis = {"calculation": {"crosstalk fl12": .4,
                                }}
    ds.config.update(analysis)
    assert "fl2_max_ctc" not in ds


def test_fl_crosstalk_priority():
    data = {"fl1_max": np.linspace(1, 1.1, 10),
            "fl2_max": np.linspace(0, 4.1, 10),
            "fl3_max": np.linspace(3, 2.5, 10),
            }
    ds = dclab.new_dataset(data)
    analysis = {"calculation": {"crosstalk fl12": .4,
                                "crosstalk fl21": .05,
                                }}
    ds.config.update(analysis)
    av = ancillaries.AncillaryFeature.available_features(ds)
    avkeys = list(av.keys())
    assert "fl1_max_ctc" in avkeys
    assert "fl2_max_ctc" in avkeys
    assert "fl3_max_ctc" not in avkeys
    reqconf = [['calculation', ['crosstalk fl21', 'crosstalk fl12']]]
    assert av["fl1_max_ctc"].req_config == reqconf
    analysis = {"calculation": {"crosstalk fl13": .1,
                                "crosstalk fl23": .7,
                                "crosstalk fl31": .2,
                                "crosstalk fl32": .2,
                                }}
    ds.config.update(analysis)
    # If there are three fl features and the corresponding crosstalk
    # values, then we must always have triple crosstalk correction.
    av2 = ancillaries.AncillaryFeature.available_features(ds)
    av2keys = list(av2.keys())
    assert "fl1_max_ctc" in av2keys
    assert "fl2_max_ctc" in av2keys
    assert "fl3_max_ctc" in av2keys
    reqconf2 = [['calculation', ["crosstalk fl21",
                                 "crosstalk fl31",
                                 "crosstalk fl12",
                                 "crosstalk fl32",
                                 "crosstalk fl13",
                                 "crosstalk fl23"]]]
    assert av2["fl1_max_ctc"].req_config == reqconf2
    assert av2["fl2_max_ctc"].req_config == reqconf2
    assert av2["fl3_max_ctc"].req_config == reqconf2


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'ancillaries.ancillary_feature.'
                            + 'BadFeatureSizeWarning')
def test_inert_ratio_cvx():
    # Brightness of the image
    ds = dclab.new_dataset(retrieve_data("rtdc_data_traces_video_bright.zip"))
    # This is something low-level and should not be done in a script.
    # Remove the brightness columns from RTDCBase to force computation with
    # the image and contour columns.
    real_ir = ds._events.pop("inert_ratio_cvx")
    # This will cause a zero-padding warning:
    comp_ir = ds["inert_ratio_cvx"]
    idcompare = ~np.isnan(comp_ir)
    # ignore first event (no image data)
    idcompare[0] = False
    assert np.allclose(real_ir[idcompare], comp_ir[idcompare])
    cleanup()


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'ancillaries.ancillary_feature.'
                            + 'BadFeatureSizeWarning')
def test_inert_ratio_prnc():
    # Brightness of the image
    ds = dclab.new_dataset(retrieve_data("rtdc_data_traces_video_bright.zip"))
    # This will cause a zero-padding warning:
    prnc = ds["inert_ratio_prnc"]
    raw = ds["inert_ratio_raw"]
    idcompare = ~np.isnan(prnc)
    # ignore first event (no image data)
    idcompare[0] = False
    diff = (prnc - raw)[idcompare]
    # only compare the first valid event which seems to be quite close
    assert np.allclose(diff[0], 0, atol=1.2e-3, rtol=0)
    cleanup()


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'ancillaries.ancillary_feature.'
                            + 'BadFeatureSizeWarning')
def test_inert_ratio_raw():
    # Brightness of the image
    ds = dclab.new_dataset(retrieve_data("rtdc_data_traces_video_bright.zip"))
    # This is something low-level and should not be done in a script.
    # Remove the brightness columns from RTDCBase to force computation with
    # the image and contour columns.
    real_ir = ds._events.pop("inert_ratio_raw")
    # This will cause a zero-padding warning:
    comp_ir = ds["inert_ratio_raw"]
    idcompare = ~np.isnan(comp_ir)
    # ignore first event (no image data)
    idcompare[0] = False
    assert np.allclose(real_ir[idcompare], comp_ir[idcompare])
    cleanup()


def test_time():
    ds = dclab.new_dataset(retrieve_data("rtdc_data_minimal.zip"))
    tt = ds["time"]
    assert tt[0] == 0
    assert np.allclose(tt[1], 0.0385)
    assert np.all(np.diff(tt) > 0)
    cleanup()


def test_volume():
    ds = dclab.new_dataset(retrieve_data("rtdc_data_minimal.zip"))
    vol = ds["volume"]
    # There are a lot of nans, because the contour is not given everywhere
    vol = vol[~np.isnan(vol)]
    assert np.allclose(vol[0], 574.60368907528346)
    assert np.allclose(vol[12], 1010.5669523203878)


if __name__ == "__main__":
    test_fl_crosstalk_2chan()
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
