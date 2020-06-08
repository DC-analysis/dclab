#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import pathlib
import time
import sys

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
                            + 'fmt_tdms.event_image.'
                            + 'InitialFrameMissingWarning')
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


@pytest.mark.filterwarnings('ignore::dclab.features.emodulus.'
                            + 'YoungsModulusLookupTableExceededWarning')
def test_emodulus_known_media():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    # legacy
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ds.config["calculation"] = {"emodulus medium": "CellCarrier",
                                "emodulus model": "elastic sphere",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    # known-media
    ds2 = dclab.new_dataset(ddict)
    ds2.config["setup"]["flow rate"] = 0.16
    ds2.config["setup"]["channel width"] = 30
    ds2.config["imaging"]["pixel size"] = .34
    ds2.config["calculation"] = {"emodulus model": "elastic sphere",
                                 "emodulus medium": "CellCarrier",
                                 "emodulus temperature": 23.0
                                 }
    assert np.sum(~np.isnan(ds["emodulus"])) > 0
    assert np.allclose(ds["emodulus"], ds2["emodulus"], equal_nan=True,
                       rtol=0, atol=1e-15)
    # ancillary feature priority check
    for af in ancillaries.AncillaryFeature.get_instances("emodulus"):
        if af.method.__name__ == "compute_emodulus_legacy":
            assert af.is_available(ds)
        else:
            assert not af.is_available(ds)
        if af.method.__name__ == "compute_emodulus_known_media":
            assert af.is_available(ds2)
        else:
            assert not af.is_available(ds2)


@pytest.mark.skipif(sys.version_info < (3, 3),
                    reason="perf_counter requires python3.3 or higher")
@pytest.mark.filterwarnings('ignore::dclab.features.emodulus.'
                            + 'YoungsModulusLookupTableExceededWarning')
def test_emodulus_legacy():
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
    t1 = time.perf_counter()
    assert "emodulus" in ds
    t2 = time.perf_counter()

    t3 = time.perf_counter()
    ds["emodulus"]
    t4 = time.perf_counter()
    assert t4 - t3 > t2 - t1


def test_emodulus_legacy_area():
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


def test_emodulus_legacy_none():
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


def test_emodulus_legacy_none2():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    assert "emodulus" not in ds, "not config for emodulus"
    ds.config["calculation"] = {"emodulus medium": "CellCarrier",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    assert "emodulus" not in ds, "emodulus model should be missing"


@pytest.mark.skipif(sys.version_info < (3, 3),
                    reason="perf_counter requires python3.3 or higher")
@pytest.mark.filterwarnings('ignore::dclab.features.emodulus.'
                            + 'YoungsModulusLookupTableExceededWarning')
def test_emodulus_legacy_viscosity_does_not_matter():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ds.config["calculation"] = {"emodulus medium": "CellCarrier",
                                "emodulus model": "elastic sphere",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5  # irrelevant
                                }
    ds2 = dclab.new_dataset(ddict)
    ds2.config["setup"]["flow rate"] = 0.16
    ds2.config["setup"]["channel width"] = 30
    ds2.config["imaging"]["pixel size"] = .34
    ds2.config["calculation"] = {"emodulus medium": "CellCarrier",
                                 "emodulus model": "elastic sphere",
                                 "emodulus temperature": 23.0,
                                 "emodulus viscosity": 0.1  # irrelevant
                                 }
    assert np.sum(~np.isnan(ds["emodulus"])) > 0
    assert np.allclose(ds["emodulus"], ds2["emodulus"], equal_nan=True,
                       rtol=0, atol=1e-15)


def test_emodulus_reservoir():
    """Reservoir measurements should not have emodulus"""
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    # legacy
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ds.config["calculation"] = {"emodulus medium": "CellCarrier",
                                "emodulus model": "elastic sphere",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    assert "emodulus" in ds
    ds2 = dclab.new_dataset(ddict)
    ds2.config["setup"]["flow rate"] = 0.16
    ds2.config["setup"]["channel width"] = 30
    ds2.config["imaging"]["pixel size"] = .34
    ds2.config["calculation"] = {"emodulus medium": "CellCarrier",
                                 "emodulus model": "elastic sphere",
                                 "emodulus temperature": 23.0,
                                 "emodulus viscosity": 0.5
                                 }
    ds2.config["setup"]["chip region"] = "reservoir"
    assert "emodulus" not in ds2


@pytest.mark.filterwarnings('ignore::dclab.features.emodulus.'
                            + 'YoungsModulusLookupTableExceededWarning')
def test_emodulus_temp_feat():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    # legacy
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ds.config["calculation"] = {"emodulus medium": "CellCarrier",
                                "emodulus model": "elastic sphere",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    ddict2 = example_data_dict(size=8472, keys=keys)
    ddict2["temp"] = 23.0 * np.ones(8472)
    # temp-feat
    ds2 = dclab.new_dataset(ddict2)
    ds2.config["setup"]["flow rate"] = 0.16
    ds2.config["setup"]["channel width"] = 30
    ds2.config["imaging"]["pixel size"] = .34
    ds2.config["calculation"] = {"emodulus medium": "CellCarrier",
                                 "emodulus model": "elastic sphere",
                                 }
    assert np.sum(~np.isnan(ds["emodulus"])) > 0
    assert np.allclose(ds["emodulus"], ds2["emodulus"], equal_nan=True,
                       rtol=0, atol=6.5e-14)
    # ancillary feature priority check
    for af in ancillaries.AncillaryFeature.get_instances("emodulus"):
        if af.method.__name__ == "compute_emodulus_legacy":
            assert af.is_available(ds)
        else:
            assert not af.is_available(ds)
        if af.method.__name__ == "compute_emodulus_temp_feat":
            assert af.is_available(ds2)
        else:
            assert not af.is_available(ds2)


@pytest.mark.filterwarnings('ignore::dclab.features.emodulus.'
                            + 'YoungsModulusLookupTableExceededWarning')
def test_emodulus_temp_feat_2():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    # legacy
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ds.config["calculation"] = {"emodulus medium": "CellCarrier",
                                "emodulus model": "elastic sphere",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    ddict2 = example_data_dict(size=8472, keys=keys)
    ddict2["temp"] = 23.0 * np.ones(8472)
    ddict2["temp"][0] = 23.5  # change first element
    # temp-feat
    ds2 = dclab.new_dataset(ddict2)
    ds2.config["setup"]["flow rate"] = 0.16
    ds2.config["setup"]["channel width"] = 30
    ds2.config["imaging"]["pixel size"] = .34
    ds2.config["calculation"] = {"emodulus medium": "CellCarrier",
                                 "emodulus model": "elastic sphere",
                                 }
    assert np.sum(~np.isnan(ds["emodulus"])) > 0
    assert np.allclose(ds["emodulus"][1:], ds2["emodulus"][1:], equal_nan=True,
                       rtol=0, atol=6.5e-14)
    assert not np.allclose(ds["emodulus"][0], ds2["emodulus"][0])
    ds3 = dclab.new_dataset(ddict)
    ds3.config["setup"]["flow rate"] = 0.16
    ds3.config["setup"]["channel width"] = 30
    ds3.config["imaging"]["pixel size"] = .34
    ds3.config["calculation"] = {"emodulus medium": "CellCarrier",
                                 "emodulus model": "elastic sphere",
                                 "emodulus temperature": 23.5,
                                 "emodulus viscosity": 0.5
                                 }
    assert np.allclose(ds3["emodulus"][0], ds2["emodulus"][0], rtol=0,
                       atol=6e-14)


@pytest.mark.filterwarnings('ignore::dclab.features.emodulus.'
                            + 'YoungsModulusLookupTableExceededWarning')
def test_emodulus_visc_only():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    # legacy
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ds.config["calculation"] = {"emodulus medium": "CellCarrier",
                                "emodulus model": "elastic sphere",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5  # irrelevant
                                }
    # visc-only
    ds2 = dclab.new_dataset(ddict)
    ds2.config["setup"]["flow rate"] = 0.16
    ds2.config["setup"]["channel width"] = 30
    ds2.config["imaging"]["pixel size"] = .34
    visc = dclab.features.emodulus.viscosity.get_viscosity(
        medium="CellCarrier",
        channel_width=30,
        flow_rate=0.16,
        temperature=23.0)
    ds2.config["calculation"] = {"emodulus model": "elastic sphere",
                                 "emodulus viscosity": visc
                                 }
    assert np.sum(~np.isnan(ds["emodulus"])) > 0
    assert np.allclose(ds["emodulus"], ds2["emodulus"], equal_nan=True,
                       rtol=0, atol=1e-15)
    # ancillary feature priority check
    for af in ancillaries.AncillaryFeature.get_instances("emodulus"):
        if af.method.__name__ == "compute_emodulus_legacy":
            assert af.is_available(ds)
        else:
            assert not af.is_available(ds)
        if af.method.__name__ == "compute_emodulus_visc_only":
            assert af.is_available(ds2)
        else:
            assert not af.is_available(ds2)


@pytest.mark.filterwarnings('ignore::dclab.features.emodulus.'
                            + 'YoungsModulusLookupTableExceededWarning')
def test_emodulus_visc_only_2():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    visc = dclab.features.emodulus.viscosity.get_viscosity(
        medium="CellCarrier",
        channel_width=30,
        flow_rate=0.16,
        temperature=23.0)
    # legacy
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ds.config["calculation"] = {"emodulus medium": "other",
                                "emodulus model": "elastic sphere",
                                "emodulus temperature": 47.0,  # irrelevant
                                "emodulus viscosity": visc
                                }
    # visc-only
    ds2 = dclab.new_dataset(ddict)
    ds2.config["setup"]["flow rate"] = 0.16
    ds2.config["setup"]["channel width"] = 30
    ds2.config["imaging"]["pixel size"] = .34
    ds2.config["calculation"] = {"emodulus model": "elastic sphere",
                                 "emodulus viscosity": visc
                                 }
    assert np.sum(~np.isnan(ds["emodulus"])) > 0
    assert np.allclose(ds["emodulus"], ds2["emodulus"], equal_nan=True,
                       rtol=0, atol=1e-15)


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


def test_ml_class_basic():
    data = {"ml_score_001": [.1, .3, .1, 0.01, .59],
            "ml_score_002": [.2, .1, .4, 0, .8],
            }
    ds = dclab.new_dataset(data)
    assert "ml_class" in ds
    assert np.allclose(ds["ml_class"], [1, 0, 1, 0, 1])
    assert issubclass(ds["ml_class"].dtype.type, np.integer)


def test_ml_class_bad_feature():
    data = {"ml_score_0-1": [.1, .3, .1, 0.01, .59],
            }
    try:
        dclab.new_dataset(data)
    except ValueError:
        pass
    else:
        assert False, "This is not a valid feature name"


def test_ml_class_bad_score_max():
    data = {"ml_score_001": [.1, .3, 99, 0.01, .59],
            "ml_score_002": [.2, .1, .4, 0, .8],
            }
    ds = dclab.new_dataset(data)
    assert "ml_class" in ds
    try:
        ds["ml_class"]
    except ValueError as e:
        assert "> 1" in e.args[0]
    else:
        assert False, "99 is not allowed"


def test_ml_class_bad_score_min():
    data = {"ml_score_001": [.1, .3, -.1, 0.01, .59],
            "ml_score_002": [.2, .1, .4, 0, .8],
            }
    ds = dclab.new_dataset(data)
    assert "ml_class" in ds
    try:
        ds["ml_class"]
    except ValueError as e:
        assert "< 0" in e.args[0]
    else:
        assert False, "negative is not allowed"


def test_ml_class_bad_score_nan():
    data = {"ml_score_001": [.1, .3, np.nan, 0.01, .59],
            "ml_score_002": [.2, .1, .4, 0, .8],
            }
    ds = dclab.new_dataset(data)
    assert "ml_class" in ds
    try:
        ds["ml_class"]
    except ValueError as e:
        assert "nan values" in e.args[0]
    else:
        assert False, "nan is not allowed"


def test_ml_class_single():
    data = {"ml_score_001": [.1, .3, .1, 0.01, .59],
            }
    ds = dclab.new_dataset(data)
    assert "ml_class" in ds
    assert np.allclose(ds["ml_class"], 0)


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
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
