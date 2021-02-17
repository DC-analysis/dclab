import pathlib
from pkg_resources import resource_filename
import sys
import tempfile
import time

import numpy as np
import pytest

import dclab
from dclab.features import emodulus
from dclab.rtdc_dataset import ancillaries

from helper_methods import example_data_dict


@pytest.mark.filterwarnings('ignore::dclab.features.emodulus.'
                            + 'YoungsModulusLookupTableExceededWarning')
def test_af_emodulus_known_media():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    # legacy
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ds.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                "emodulus medium": "CellCarrier",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    # known-media
    ds2 = dclab.new_dataset(ddict)
    ds2.config["setup"]["flow rate"] = 0.16
    ds2.config["setup"]["channel width"] = 30
    ds2.config["imaging"]["pixel size"] = .34
    ds2.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                 "emodulus medium": "CellCarrier",
                                 "emodulus temperature": 23.0
                                 }
    assert np.sum(~np.isnan(ds["emodulus"])) > 0
    assert np.allclose(ds["emodulus"], ds2["emodulus"], equal_nan=True,
                       rtol=0, atol=1e-15)
    # ancillary feature priority check
    for af in ancillaries.AncillaryFeature.get_instances("emodulus"):
        if af.data == "DEPRECATED":
            continue
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
def test_af_emodulus_legacy():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ds.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                "emodulus medium": "CellCarrier",
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


def test_af_emodulus_legacy_area():
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
    ds.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                "emodulus medium": "CellCarrier",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    assert "emodulus" in ds


def test_af_emodulus_legacy_none():
    keys = ["area_msd", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    assert "emodulus" not in ds, "not config for emodulus"
    ds.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                "emodulus medium": "CellCarrier",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    assert "emodulus" not in ds, "column 'area_um' should be missing"


def test_af_emodulus_legacy_none2():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    assert "emodulus" not in ds, "not config for emodulus"
    ds.config["calculation"] = {"emodulus medium": "CellCarrier",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    assert "emodulus" not in ds, "emodulus lut should be missing"


@pytest.mark.filterwarnings('ignore::dclab.features.emodulus.'
                            + 'YoungsModulusLookupTableExceededWarning')
def test_af_emodulus_legacy_viscosity_does_not_matter():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ds.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                "emodulus medium": "CellCarrier",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5  # irrelevant
                                }
    ds2 = dclab.new_dataset(ddict)
    ds2.config["setup"]["flow rate"] = 0.16
    ds2.config["setup"]["channel width"] = 30
    ds2.config["imaging"]["pixel size"] = .34
    ds2.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                 "emodulus medium": "CellCarrier",
                                 "emodulus temperature": 23.0,
                                 "emodulus viscosity": 0.1  # irrelevant
                                 }
    assert np.sum(~np.isnan(ds["emodulus"])) > 0
    assert np.allclose(ds["emodulus"], ds2["emodulus"], equal_nan=True,
                       rtol=0, atol=1e-15)


def test_af_emodulus_reservoir():
    """Reservoir measurements should not have emodulus"""
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    # legacy
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ds.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                "emodulus medium": "CellCarrier",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    assert "emodulus" in ds
    ds2 = dclab.new_dataset(ddict)
    ds2.config["setup"]["flow rate"] = 0.16
    ds2.config["setup"]["channel width"] = 30
    ds2.config["imaging"]["pixel size"] = .34
    ds2.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                 "emodulus medium": "CellCarrier",
                                 "emodulus temperature": 23.0,
                                 "emodulus viscosity": 0.5
                                 }
    ds2.config["setup"]["chip region"] = "reservoir"
    assert "emodulus" not in ds2


@pytest.mark.filterwarnings('ignore::dclab.features.emodulus.'
                            + 'YoungsModulusLookupTableExceededWarning')
def test_af_emodulus_temp_feat():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    # legacy
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ds.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                "emodulus medium": "CellCarrier",
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
    ds2.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                 "emodulus medium": "CellCarrier",
                                 }
    assert np.sum(~np.isnan(ds["emodulus"])) > 0
    assert np.allclose(ds["emodulus"], ds2["emodulus"], equal_nan=True,
                       rtol=0, atol=6.5e-14)
    # ancillary feature priority check
    for af in ancillaries.AncillaryFeature.get_instances("emodulus"):
        if af.data == "DEPRECATED":
            continue
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
def test_af_emodulus_temp_feat_2():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    # legacy
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ds.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                "emodulus medium": "CellCarrier",
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
                                 "emodulus lut": "LE-2D-FEM-19",
                                 }
    assert np.sum(~np.isnan(ds["emodulus"])) > 0
    assert np.allclose(ds["emodulus"][1:], ds2["emodulus"][1:], equal_nan=True,
                       rtol=0, atol=6.5e-14)
    assert not np.allclose(ds["emodulus"][0], ds2["emodulus"][0])
    ds3 = dclab.new_dataset(ddict)
    ds3.config["setup"]["flow rate"] = 0.16
    ds3.config["setup"]["channel width"] = 30
    ds3.config["imaging"]["pixel size"] = .34
    ds3.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                 "emodulus medium": "CellCarrier",
                                 "emodulus temperature": 23.5,
                                 "emodulus viscosity": 0.5
                                 }
    assert np.allclose(ds3["emodulus"][0], ds2["emodulus"][0], rtol=0,
                       atol=6e-14)


@pytest.mark.filterwarnings('ignore::dclab.features.emodulus.'
                            + 'YoungsModulusLookupTableExceededWarning')
def test_af_emodulus_visc_only():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    # legacy
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ds.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                "emodulus medium": "CellCarrier",
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
    ds2.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                 "emodulus viscosity": visc
                                 }
    assert np.sum(~np.isnan(ds["emodulus"])) > 0
    assert np.allclose(ds["emodulus"], ds2["emodulus"], equal_nan=True,
                       rtol=0, atol=1e-15)
    # ancillary feature priority check
    for af in ancillaries.AncillaryFeature.get_instances("emodulus"):
        if af.data == "DEPRECATED":
            continue
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
def test_af_emodulus_visc_only_2():
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
    ds.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                "emodulus medium": "other",
                                "emodulus temperature": 47.0,  # irrelevant
                                "emodulus viscosity": visc
                                }
    # visc-only
    ds2 = dclab.new_dataset(ddict)
    ds2.config["setup"]["flow rate"] = 0.16
    ds2.config["setup"]["channel width"] = 30
    ds2.config["imaging"]["pixel size"] = .34
    ds2.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                 "emodulus viscosity": visc
                                 }
    assert np.sum(~np.isnan(ds["emodulus"])) > 0
    assert np.allclose(ds["emodulus"], ds2["emodulus"], equal_nan=True,
                       rtol=0, atol=1e-15)


def test_bad_lut_data():
    try:
        emodulus.load_lut("bad_string_asdkubhasd")
    except ValueError:
        pass
    else:
        assert False, "Invalid `lut_data` results in ValueError"


@pytest.mark.filterwarnings(
    'ignore::dclab.features.emodulus.KnowWhatYouAreDoingWarning')
def test_extrapolate():
    """Test whether spline interpolation gives reasonable results"""
    lut, _ = emodulus.load_lut("FEM-2Daxis")

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
            datax=area_um,
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
    assert valid_ones == 149


def test_load_lut_from_array():
    ref_lut, ref_meta = emodulus.load_lut("FEM-2Daxis")
    lut2, meta2 = emodulus.load_lut((ref_lut, ref_meta))
    assert np.all(ref_lut == lut2)
    assert ref_meta == meta2
    assert ref_lut is not lut2, "data should be copied"
    assert ref_meta is not meta2, "meta data should be copied"


def test_load_lut_from_path():
    ref_lut, ref_meta = emodulus.load_lut("FEM-2Daxis")
    path = resource_filename("dclab.features.emodulus",
                             emodulus.load.INTERNAL_LUTS["LE-2D-FEM-19"])
    lut2, meta2 = emodulus.load_lut(path)
    assert np.all(ref_lut == lut2)
    assert ref_meta == meta2


def test_load_lut_from_badobject():
    try:
        emodulus.load_lut({"test": "nonesense"})
    except ValueError:
        pass
    else:
        assert False, "dict should not be supported"


def test_load_lut_from_badpath():
    try:
        emodulus.load_lut("peter/pan.txt")
    except ValueError:
        pass
    else:
        assert False, "dict should not be supported"


def test_pixelation_correction_volume():
    ddelt = emodulus.get_pixelation_delta(feat_corr="deform",
                                          feat_absc="volume",
                                          data_absc=100,
                                          px_um=0.34)
    assert np.allclose(ddelt, 0.011464479831134636)


def test_register_external_lut():
    """Load an external LUT and compute YM data"""
    identifier = "test-test_register_external_lut"
    path = resource_filename("dclab.features.emodulus",
                             emodulus.load.INTERNAL_LUTS["LE-2D-FEM-19"])
    emodulus.register_lut(path, identifier=identifier)
    # cleanup
    emodulus.load.EXTERNAL_LUTS.clear()


@pytest.mark.filterwarnings('ignore::dclab.features.emodulus.'
                            + 'YoungsModulusLookupTableExceededWarning')
def test_register_external_lut_and_get_emodulus():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    # from internal LUT
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ds.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                "emodulus medium": "CellCarrier",
                                "emodulus temperature": 23.0
                                }
    assert np.sum(~np.isnan(ds["emodulus"])) > 0
    # from external LUT
    identifier = "test-test_register_external_lut"
    path = resource_filename("dclab.features.emodulus",
                             emodulus.load.INTERNAL_LUTS["LE-2D-FEM-19"])
    emodulus.register_lut(path, identifier=identifier)
    ds2 = dclab.new_dataset(ddict)
    ds2.config["setup"]["flow rate"] = 0.16
    ds2.config["setup"]["channel width"] = 30
    ds2.config["imaging"]["pixel size"] = .34
    ds2.config["calculation"] = {"emodulus lut": identifier,
                                 "emodulus medium": "CellCarrier",
                                 "emodulus temperature": 23.0
                                 }
    assert np.sum(~np.isnan(ds2["emodulus"])) > 0
    assert np.allclose(ds["emodulus"], ds2["emodulus"], equal_nan=True,
                       rtol=0, atol=1e-15)


@pytest.mark.filterwarnings('ignore::dclab.features.emodulus.'
                            + 'YoungsModulusLookupTableExceededWarning')
def test_register_external_lut_with_internal_identifier():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    # from internal LUT
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ds.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                "emodulus medium": "CellCarrier",
                                "emodulus temperature": 23.0
                                }
    assert np.sum(~np.isnan(ds["emodulus"])) > 0
    # from external LUT
    path = resource_filename("dclab.features.emodulus",
                             emodulus.load.INTERNAL_LUTS["LE-2D-FEM-19"])
    path2 = pathlib.Path(tempfile.mkdtemp("external_lut_with_id")) / "lut.txt"
    text = pathlib.Path(path).read_text().split("\n")
    text[24] = '#   "identifier": "LE-2D-FEM-19b",'
    path2.write_text("\n".join(text))
    emodulus.register_lut(path2)


@pytest.mark.filterwarnings('ignore::dclab.features.emodulus.'
                            + 'YoungsModulusLookupTableExceededWarning')
def test_register_external_lut_without_identifier():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    # from internal LUT
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ds.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                "emodulus medium": "CellCarrier",
                                "emodulus temperature": 23.0
                                }
    assert np.sum(~np.isnan(ds["emodulus"])) > 0
    # from external LUT
    path = resource_filename("dclab.features.emodulus",
                             emodulus.load.INTERNAL_LUTS["LE-2D-FEM-19"])
    with pytest.raises(ValueError):
        emodulus.register_lut(path)


@pytest.mark.filterwarnings('ignore::dclab.features.emodulus.'
                            + 'YoungsModulusLookupTableExceededWarning')
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
    assert np.allclose(emod[50, 50], 0.5527978323425833)
    assert np.allclose(emod[80, 50], 0.45677680242729324)

    assert np.allclose(emod[10, 80], 1.5744560306483262)
    assert np.allclose(emod[50, 80], 0.73534561544655519)
    assert np.allclose(emod[80, 80], 0.60737083178222251)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
