import numpy as np

import pytest

import dclab
from dclab.features.fl_crosstalk import correct_crosstalk
from dclab.rtdc_dataset import ancillaries

from helper_methods import retrieve_data


def test_af_fl_crosstalk_2chan():
    pytest.importorskip("nptdms")
    ds = dclab.new_dataset(retrieve_data("fmt-tdms_2fl-no-image_2017.zip"))
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


def test_af_fl_crosstalk_3chanvs2chan():
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


def test_af_fl_crosstalk_missing():
    data = {"fl1_max": np.linspace(1, 1.1, 10),
            "fl2_max": np.linspace(0, 4.1, 10),
            }
    ds = dclab.new_dataset(data)
    analysis = {"calculation": {"crosstalk fl12": .4,
                                }}
    ds.config.update(analysis)
    assert "fl2_max_ctc" not in ds


def test_af_fl_crosstalk_priority():
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
