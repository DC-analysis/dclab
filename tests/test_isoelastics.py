import pathlib
import warnings

import pytest

import numpy as np

import dclab
from dclab import isoelastics as iso
from dclab.features import emodulus
from dclab.features.emodulus import pxcorr

from helper_methods import example_data_dict


def get_isofile(name="example_isoelastics.txt"):
    thisdir = pathlib.Path(__file__).parent
    return thisdir / "data" / name


def test_bad_isoelastic_undefined_feature():
    """bad feature"""
    i1 = iso.Isoelastics([get_isofile()])
    with pytest.raises(KeyError, match="area_ratio"):
        i1.get(col1="deform",
               col2="area_ratio",
               lut_identifier="test-LE-2D-ana-18",
               channel_width=20,
               flow_rate=0.04,
               viscosity=15,
               add_px_err=False,
               px_um=0.34)


def test_bad_isoelastic_undefined_lut_for_data():
    """Bad LUT identifier in isoelastics instance"""
    i1 = iso.Isoelastics([get_isofile()])
    with pytest.raises(KeyError, match="LE-2D-FEM-19"):
        i1.get(col1="deform",
               col2="area_um",
               lut_identifier="LE-2D-FEM-19",
               channel_width=20,
               flow_rate=0.04,
               viscosity=15,
               add_px_err=False,
               px_um=0.34)


def test_bad_isoelastic_unknown_feature():
    i1 = iso.Isoelastics([get_isofile()])
    with pytest.raises(ValueError, match="bad_feature"):
        i1.get(col1="deform",
               col2="bad_feature",
               lut_identifier="LE-2D-FEM-19",
               channel_width=20,
               flow_rate=0.04,
               viscosity=15,
               add_px_err=False,
               px_um=0.34)


def test_bad_isoelastic_unknown_lut_identifier():
    i1 = iso.Isoelastics([get_isofile()])
    with pytest.raises(KeyError, match="LE-2D-FEM-99-nonexistent"):
        i1.get(col1="deform",
               col2="area_um",
               lut_identifier="LE-2D-FEM-99-nonexistent",
               channel_width=20,
               flow_rate=0.04,
               viscosity=15,
               add_px_err=False,
               px_um=0.34)


def test_circ():
    i1 = iso.Isoelastics([get_isofile()])
    iso1 = i1._data["test-LE-2D-ana-18"]["area_um"]["deform"]["isoelastics"]
    iso2 = i1._data["test-LE-2D-ana-18"]["area_um"]["circ"]["isoelastics"]
    assert np.allclose(iso1[0][:, 1], 1 - iso2[0][:, 1])


def test_circ_get():
    i1 = iso.Isoelastics([get_isofile()])
    iso_circ = i1.get(col1="area_um",
                      col2="circ",
                      lut_identifier="test-LE-2D-ana-18",
                      channel_width=15,
                      flow_rate=0.04,
                      viscosity=15)
    iso_deform = i1.get(col1="area_um",
                        col2="deform",
                        lut_identifier="test-LE-2D-ana-18",
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
    isoel = i1._data["test-LE-2D-ana-18"]["area_um"]["deform"]["isoelastics"]
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
                   lut_identifier="test-LE-2D-ana-18",
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
                   col2="area_ratio",
                   **kwargs)
    except KeyError:
        pass
    except BaseException:
        raise
    else:
        assert False, "undefined column volume"


def test_data_slicing():
    i1 = iso.Isoelastics([get_isofile()])
    iso1 = i1._data["test-LE-2D-ana-18"]["area_um"]["deform"]["isoelastics"]
    iso2 = i1._data["test-LE-2D-ana-18"]["deform"]["area_um"]["isoelastics"]
    for ii in range(len(iso1)):
        assert np.all(iso1[ii][:, 2] == iso2[ii][:, 2])
        assert np.all(iso1[ii][:, 0] == iso2[ii][:, 1])
        assert np.all(iso1[ii][:, 1] == iso2[ii][:, 0])


def test_data_structure():
    i1 = iso.Isoelastics([get_isofile()])
    # basic import
    assert "test-LE-2D-ana-18" in i1._data
    assert "deform" in i1._data["test-LE-2D-ana-18"]
    assert "area_um" in i1._data["test-LE-2D-ana-18"]["deform"]
    assert "area_um" in i1._data["test-LE-2D-ana-18"]
    assert "deform" in i1._data["test-LE-2D-ana-18"]["area_um"]
    # circularity
    assert "circ" in i1._data["test-LE-2D-ana-18"]
    assert "area_um" in i1._data["test-LE-2D-ana-18"]["circ"]
    assert "area_um" in i1._data["test-LE-2D-ana-18"]
    assert "circ" in i1._data["test-LE-2D-ana-18"]["area_um"]
    # metadata
    meta1 = i1._data["test-LE-2D-ana-18"]["area_um"]["deform"]["meta"]
    meta2 = i1._data["test-LE-2D-ana-18"]["deform"]["area_um"]["meta"]
    assert meta1 == meta2


def test_get():
    i1 = iso.Isoelastics([get_isofile()])
    data = i1.get(col1="area_um",
                  col2="deform",
                  channel_width=20,
                  flow_rate=0.04,
                  viscosity=15,
                  lut_identifier="test-LE-2D-ana-18")
    refd = i1._data["test-LE-2D-ana-18"]["area_um"]["deform"]["isoelastics"]

    for a, b in zip(data, refd):
        assert np.all(a == b)


def test_pixel_err():
    i1 = iso.Isoelastics([get_isofile()])
    isoel = i1._data["test-LE-2D-ana-18"]["area_um"]["deform"]["isoelastics"]
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
        iss[:, 1] -= pxcorr.corr_deform_with_area_um(area_um=iss[:, 0],
                                                     px_um=px_um)
        isoel_corr.append(iss)

    for ii in range(len(isoel)):
        assert not np.allclose(isoel[ii], isoel_err[ii])
        assert np.allclose(isoel[ii], isoel_corr[ii])

    try:
        i1.add_px_err(isoel=isoel,
                      col1="deform",
                      col2="deform",
                      px_um=px_um,
                      inplace=False)
    except ValueError:
        pass
    else:
        assert False, "identical columns"

    try:
        i1.add_px_err(isoel=isoel,
                      col1="deform",
                      col2="circ",
                      px_um=px_um,
                      inplace=False)
    except KeyError:
        pass
    except BaseException:
        raise
    else:
        assert False, "area_um required"


def test_volume_basic():
    """Reproduce exact data from simulation result"""
    i1 = iso.get_default()
    data = i1.get(col1="volume",
                  col2="deform",
                  channel_width=20,
                  flow_rate=0.04,
                  viscosity=15,
                  lut_identifier="LE-2D-FEM-19",
                  add_px_err=False,
                  px_um=None)
    assert np.allclose(data[0][0], [1.60502e+02, 4.38040e-02, 1.06000e+00])
    assert np.allclose(data[0][-1], [4.32113e+02, 1.37544e-01, 1.06000e+00])
    assert np.allclose(data[1][0], [1.61559e+02, 2.60143e-02, 1.34000e+00])
    assert np.allclose(data[-1][-1], [3.18845e+03, 1.45521e-02, 1.01800e+01])


def test_volume_pxcorr():
    """Deformation is pixelation-corrected using volume"""
    i1 = iso.get_default()
    data = i1.get(col1="volume",
                  col2="deform",
                  channel_width=20,
                  flow_rate=None,
                  viscosity=None,
                  lut_identifier="LE-2D-FEM-19",
                  add_px_err=True,
                  px_um=0.34)
    ddelt = pxcorr.corr_deform_with_volume(1.60502e+02, px_um=0.34)
    assert np.allclose(data[0][0], [1.60502e+02,
                                    4.38040e-02 + ddelt,
                                    1.06000e+00])


def test_volume_scale():
    """Simple volume scale"""
    i1 = iso.get_default()
    data = i1.get(col1="volume",
                  col2="deform",
                  channel_width=25,
                  flow_rate=0.04,
                  viscosity=15,
                  lut_identifier="LE-2D-FEM-19",
                  add_px_err=False,
                  px_um=None)

    assert np.allclose(data[0][0], [1.60502e+02 * (25 / 20)**3,
                                    4.38040e-02,
                                    1.06000e+00 * (20 / 25)**3])


def test_volume_scale_2():
    """The default values are used if set to None"""
    i1 = iso.get_default()
    data = i1.get(col1="volume",
                  col2="deform",
                  channel_width=25,
                  flow_rate=None,
                  viscosity=None,
                  lut_identifier="LE-2D-FEM-19",
                  add_px_err=False,
                  px_um=None)
    assert np.allclose(data[0][0], [1.60502e+02 * (25 / 20)**3,
                                    4.38040e-02,
                                    1.06000e+00 * (20 / 25)**3])


def test_volume_switch():
    """Switch the columns"""
    i1 = iso.get_default()
    data = i1.get(col1="deform",
                  col2="volume",
                  channel_width=20,
                  flow_rate=0.04,
                  viscosity=15,
                  lut_identifier="LE-2D-FEM-19",
                  add_px_err=False,
                  px_um=None)
    assert np.allclose(data[0][0], [4.38040e-02, 1.60502e+02, 1.06000e+00])
    assert np.allclose(data[-1][-1], [1.45521e-02, 3.18845e+03, 1.01800e+01])


def test_volume_switch_scale():
    """Switch the columns and change the scale"""
    i1 = iso.get_default()
    data = i1.get(col1="deform",
                  col2="volume",
                  channel_width=25,
                  flow_rate=0.04,
                  viscosity=15,
                  lut_identifier="LE-2D-FEM-19",
                  add_px_err=False,
                  px_um=None)
    assert np.allclose(data[0][0], [4.38040e-02,
                                    1.60502e+02 * (25 / 20)**3,
                                    1.06000e+00 * (20 / 25)**3])
    assert np.allclose(data[-1][-1], [1.45521e-02,
                                      3.18845e+03 * (25 / 20)**3,
                                      1.01800e+01 * (20 / 25)**3])


def test_with_rtdc():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    # legacy
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["setup"]["temperature"] = 23.0
    ds.config["setup"]["medium"] = "CellCarrier"
    ds.config["imaging"]["pixel size"] = .34
    i1 = iso.get_default()
    data1 = i1.get_with_rtdcbase(col1="area_um",
                                 col2="deform",
                                 lut_identifier="LE-2D-FEM-19",
                                 dataset=ds,
                                 viscosity=None,
                                 add_px_err=False)

    viscosity = emodulus.viscosity.get_viscosity(
        medium="CellCarrier",
        channel_width=ds.config["setup"]["channel width"],
        flow_rate=ds.config["setup"]["flow rate"],
        temperature=ds.config["setup"]["temperature"],
        model='buyukurganci-2022')
    data2 = i1.get(col1="area_um",
                   col2="deform",
                   lut_identifier="LE-2D-FEM-19",
                   channel_width=ds.config["setup"]["channel width"],
                   flow_rate=ds.config["setup"]["flow rate"],
                   viscosity=viscosity,
                   add_px_err=False,
                   px_um=ds.config["imaging"]["pixel size"])
    for d1, d2 in zip(data1, data2):
        assert np.allclose(d1, d2, atol=0, rtol=1e-14)


def test_with_rtdc_warning():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    # legacy
    ds = dclab.new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["setup"]["medium"] = "CellCarrier"
    ds.config["imaging"]["pixel size"] = .34
    i1 = iso.get_default()
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning (temperature missing).
        i1.get_with_rtdcbase(col1="area_um",
                             col2="deform",
                             lut_identifier="LE-2D-FEM-19",
                             dataset=ds,
                             viscosity=None,
                             add_px_err=False)
        # Verify some things
        assert len(w) == 1
        assert issubclass(w[-1].category,
                          iso.IsoelasticsEmodulusMeaninglessWarning)
        assert "plotting" in str(w[-1].message)
