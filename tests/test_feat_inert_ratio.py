import numpy as np
import pytest

import dclab
from dclab import new_dataset
from dclab.features import inert_ratio as ir

from helper_methods import retrieve_data


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'ancillaries.ancillary_feature.'
                            + 'BadFeatureSizeWarning')
def test_af_inert_ratio_cvx():
    pytest.importorskip("nptdms")
    # Brightness of the image
    ds = dclab.new_dataset(retrieve_data("fmt-tdms_fl-image-bright_2017.zip"))
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


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'ancillaries.ancillary_feature.'
                            + 'BadFeatureSizeWarning')
def test_af_inert_ratio_prnc():
    pytest.importorskip("nptdms")
    # Brightness of the image
    ds = dclab.new_dataset(retrieve_data("fmt-tdms_fl-image-bright_2017.zip"))
    # This will cause a zero-padding warning:
    prnc = ds["inert_ratio_prnc"]
    raw = ds["inert_ratio_raw"]
    idcompare = ~np.isnan(prnc)
    # ignore first event (no image data)
    idcompare[0] = False
    diff = (prnc - raw)[idcompare]
    # only compare the first valid event which seems to be quite close
    assert np.allclose(diff[0], 0, atol=1.2e-3, rtol=0)


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'ancillaries.ancillary_feature.'
                            + 'BadFeatureSizeWarning')
def test_af_inert_ratio_raw():
    pytest.importorskip("nptdms")
    # Brightness of the image
    ds = dclab.new_dataset(retrieve_data("fmt-tdms_fl-image-bright_2017.zip"))
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


def test_inert_ratio_raw():
    ds = new_dataset(retrieve_data("fmt-hdf5_mask-contour_2018.zip"))

    raw = ir.get_inert_ratio_raw(cont=ds["contour"])
    ref = np.array([4.25854232,  1.22342663,  4.64971179,  1.70914857,
                    3.62797492, 1.51502192,  2.74757573,  1.79841136])
    assert np.allclose(ref, raw, rtol=0, atol=5e-9)


def test_inert_ratio_prnc():
    """Test equivalence of inert_ratio_raw and inert_ratio_prnc"""
    t = np.linspace(0, 2*np.pi, 300)

    x1 = 1.7 * np.cos(t)
    y1 = 1.1 * np.sin(t)
    c1 = np.dstack((x1, y1))[0]

    phi = np.arctan2(y1, x1)
    rho = np.sqrt(x1**2 + y1**2)

    for theta in np.linspace(0.1, 2*np.pi, 14):  # arbitrary rotation
        for pos_x in np.linspace(-5, 20, 8):  # arbitrary x shift
            for pos_y in np.linspace(-4.6, 17, 4):  # arbitrary y shift
                x2 = rho * np.cos(phi + theta) + pos_x
                y2 = rho * np.sin(phi + theta) + pos_y

                c2 = np.dstack((x2, y2))[0]
                raw = ir.get_inert_ratio_raw(c1)
                prnc = ir.get_inert_ratio_prnc(c2)

                assert np.allclose(raw, prnc, rtol=0, atol=1e-10)


def test_inert_ratio_prnc_simple_1():
    c = np.array([[0, 0],
                  [0, 1],
                  [0, 2],
                  [1, 2],
                  [2, 2],
                  [3, 2],
                  [3, 1],
                  [3, 0],
                  [2, 0],
                  [1, 0],
                  [0, 0]])
    raw = ir.get_inert_ratio_raw(c)
    prnc = ir.get_inert_ratio_prnc(c)
    tilt = ir.get_tilt(c)
    assert np.allclose(raw, 1.5)
    assert np.allclose(prnc, 1.5)
    assert np.allclose(tilt, 0)


def test_inert_ratio_prnc_simple_2():
    c = np.array([[0, 0],
                  [1, 1],
                  [2, 2],
                  [3, 3],
                  [4, 2],
                  [5, 1],
                  [4, 0],
                  [3, -1],
                  [2, -2],
                  [1, -1],
                  [0, 0]])
    raw = ir.get_inert_ratio_raw(c)
    prnc = ir.get_inert_ratio_prnc(c)
    tilt = ir.get_tilt(c)
    assert np.allclose(raw, 1)
    assert np.allclose(prnc, 1.5)
    assert np.allclose(tilt, np.pi/4)


def test_tilt():
    t = np.linspace(0, 2*np.pi, 300)

    x1 = 1.7 * np.cos(t)
    y1 = 1.1 * np.sin(t)

    phi = np.arctan2(y1, x1)
    rho = np.sqrt(x1**2 + y1**2)

    for theta in np.linspace(-.3, 2.2*np.pi, 32):  # arbitrary rotation
        x2 = rho * np.cos(phi + theta)
        y2 = rho * np.sin(phi + theta)

        c2 = np.dstack((x2, y2))[0]
        tilt = ir.get_tilt(c2)

        th = np.mod(theta, np.pi)
        if th > np.pi/2:
            th = np.pi - th
        assert np.allclose(tilt, th)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
