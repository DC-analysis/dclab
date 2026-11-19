import threading

import numpy as np
import pytest

import dclab
from dclab.kde import smooth_contour
from dclab.kde.base import ContourSpacingTooLarge


def test_compute_contour_opening_angles():
    contour = [
        [0, 0],
        [1, 0],
        [0.5, np.sqrt(3)/2]
    ]

    angles = smooth_contour.compute_contour_opening_angles(
        contour=contour,
        xrange=[0, 1],
        yrange=[0, 1],
        xscale="linear",
        yscale="linear",
    )
    assert np.allclose(angles, np.pi/3)


def test_compute_contour_opening_angles_shifted():
    contour = [
        [0, 0],
        [1, 0],
        [0.5, np.sqrt(3)/2]
    ]
    contour = np.array(contour) + 1
    angles = smooth_contour.compute_contour_opening_angles(
        contour=contour,
        xrange=[0, 1],
        yrange=[0, 1],
        xscale="linear",
        yscale="linear",
    )
    assert np.allclose(angles, np.pi/3)


def test_compute_contour_opening_angles_advanced():
    contour = [
        [0, 0],
        [0.5, np.sqrt(3) / 2],
        [1, 0],
        [1.5, np.sqrt(3) / 2],
        [0, np.sqrt(3) / 2],
        [0, 0],
    ]
    expected = [np.pi/6, np.pi/3, np.pi/3, np.pi/3, np.pi/2]
    angles = smooth_contour.compute_contour_opening_angles(
        contour=contour,
        xrange=[0, 1],
        yrange=[0, 1],
        xscale="linear",
        yscale="linear",
    )
    assert np.allclose(angles, expected)


def test_compute_contour_opening_angles_log_scale():
    contour = [
        [0, 0],
        [1, 0],
        [0.5, np.sqrt(3)/2]
    ]
    contour = 10**(np.array(contour) + 1)
    # The contour is normalized to the plot range on a log scale,
    # i.e. log10 of the range [10, 100] is [1, 2].
    angles = smooth_contour.compute_contour_opening_angles(
        contour=contour,
        xrange=[10, 100],
        yrange=[10, 100],
        xscale="log",
        yscale="log",
    )
    assert np.allclose(angles, np.pi/3)


def test_compute_contour_opening_angles_log_scale_outside_range():
    """Contour points outside the plot range have valid angles"""
    contour = [
        [0, 0],
        [1, 0],
        [0.5, np.sqrt(3)/2]
    ]
    contour = 10**(np.array(contour) + 1)
    # The plot range does not include the entire contour. Before, the
    # contour was normalized linearly and then log10 was applied, which
    # resulted in NaN values for points left of the plot range.
    angles = smooth_contour.compute_contour_opening_angles(
        contour=contour,
        xrange=[20, 100],
        yrange=[10, 100],
        xscale="log",
        yscale="log",
    )
    assert np.all(np.isfinite(angles))


def test_compute_contour_opening_angles_zero():
    contour = [
        [0, 0],
        [0, 0],
        [1, 0],
        [0.5, np.sqrt(3)/2]
    ]

    angles = smooth_contour.compute_contour_opening_angles(
        contour=contour,
        xrange=[0, 1],
        yrange=[0, 1],
        xscale="linear",
        yscale="linear",
    )

    # cannot compute opening angle for point on point
    assert np.isnan(angles[0])
    assert np.isnan(angles[1])
    assert np.allclose(angles[2], np.pi/3)
    assert np.allclose(angles[3], np.pi/3)


def test_find_smooth_contour_spacing_max_length():
    np.random.seed(47)
    x0 = np.random.normal(loc=100, scale=10, size=100)
    y0 = np.random.normal(loc=.1, scale=.01, size=100)

    ds = dclab.new_dataset({"area_um": x0, "deform": y0})

    res = smooth_contour.find_smooth_contour_spacing(
        ds_list=[ds],
        xax="area_um",
        yax="deform",
        xrange=(50, 150),
        yrange=(0.09, 0.11),
        quantiles=[0.90],
    )

    assert res["total iterations"] == 3
    assert res["success"]
    assert not res["corners found"]
    assert res["reason"] == "maximum contour length reached"
    assert np.allclose(res["spacing x"], 0.28099409345400594,
                       rtol=0, atol=1e-12)
    assert np.allclose(res["spacing y"], 0.00014034229940989058,
                       rtol=0, atol=1e-12)


def test_find_smooth_contour_spacing_dataset_all_filtered():
    np.random.seed(47)
    x0 = np.random.normal(loc=100, scale=10, size=100)
    y0 = np.random.normal(loc=.1, scale=.01, size=100)

    ds = dclab.new_dataset({"area_um": x0, "deform": y0})
    ds_filtered = dclab.new_dataset({"area_um": x0, "deform": y0})
    ds_filtered.filter.manual[:] = False
    ds_filtered.apply_filter()

    kwargs = {"xax": "area_um",
              "yax": "deform",
              "xrange": (50, 150),
              "yrange": (0.09, 0.11),
              "quantiles": [0.90],
              }
    res = smooth_contour.find_smooth_contour_spacing(
        ds_list=[ds], **kwargs)
    # The dataset without events must be ignored (was an IndexError).
    with pytest.warns(ContourSpacingTooLarge,
                      match="Contour not possible"):
        res2 = smooth_contour.find_smooth_contour_spacing(
            ds_list=[ds_filtered, ds], **kwargs)
    assert res == res2


def test_find_smooth_contour_spacing_no_contour():
    np.random.seed(47)
    x0 = np.random.normal(loc=100, scale=10, size=100)
    y0 = np.random.normal(loc=.1, scale=.01, size=100)

    ds = dclab.new_dataset({"area_um": x0, "deform": y0})
    ds.filter.manual[:] = False
    ds.apply_filter()

    with pytest.warns(ContourSpacingTooLarge,
                      match="Contour not possible"):
        res = smooth_contour.find_smooth_contour_spacing(
            ds_list=[ds],
            xax="area_um",
            yax="deform",
            xrange=(50, 150),
            yrange=(0.09, 0.11),
            quantiles=[0.90],
        )
    assert res["total iterations"] == 1
    assert not res["success"]
    assert res["reason"] == "no contour found"


def test_find_smooth_contour_spacing_max_iter():
    np.random.seed(47)
    x0 = np.random.normal(loc=100, scale=10, size=5)
    y0 = np.random.normal(loc=.1, scale=.01, size=5)

    ds = dclab.new_dataset({"area_um": x0, "deform": y0})

    res = smooth_contour.find_smooth_contour_spacing(
        ds_list=[ds],
        xax="area_um",
        yax="deform",
        xrange=(50, 150),
        yrange=(0.09, 0.11),
        quantiles=[0.99],
        max_iter=1,
    )

    assert res["total iterations"] == 1
    assert not res["success"]
    assert not res["corners found"]
    assert res["reason"] == "maximum iterations reached"
    assert np.allclose(res["spacing x"], 0.46201586550424045,
                       rtol=0, atol=1e-12)
    assert np.allclose(res["spacing y"], 0.00024627719927894213,
                       rtol=0, atol=1e-12)


def test_find_smooth_contour_spacing_opening_angle():
    np.random.seed(47)
    x0 = np.random.normal(loc=100, scale=10, size=10000)
    y0 = np.random.normal(loc=.1, scale=.01, size=10000)

    ds = dclab.new_dataset({"area_um": x0, "deform": y0})

    res = smooth_contour.find_smooth_contour_spacing(
        ds_list=[ds],
        xax="area_um",
        yax="deform",
        xrange=(50, 150),
        yrange=(0.09, 0.11),
        quantiles=[0.90],
    )

    assert res["total iterations"] == 3
    assert res["success"]
    assert not res["corners found"]
    assert res["reason"] == "target opening angle reached"
    assert np.allclose(res["spacing x"], 0.27646847583098244,
                       rtol=0, atol=1e-12)
    assert np.allclose(res["spacing y"], 0.00016320865830594234,
                       rtol=0, atol=1e-12)


def test_find_smooth_contour_spacing_corners():
    np.random.seed(47)
    x0 = np.random.normal(loc=100, scale=10, size=10000)
    y0 = np.random.normal(loc=.1, scale=.01, size=10000)

    # force corners to appear
    used = x0 > 100

    ds = dclab.new_dataset({"area_um": x0[used], "deform": y0[used]})

    res = smooth_contour.find_smooth_contour_spacing(
        ds_list=[ds],
        xax="area_um",
        yax="deform",
        xrange=(50, 150),
        yrange=(0.09, 0.11),
        quantiles=[0.90],
    )

    assert res["total iterations"] == 3
    assert res["success"]
    assert res["reason"] == "maximum contour length reached"
    assert res["corners found"]
    assert np.allclose(res["spacing x"], 0.16515957250959012,
                       rtol=0, atol=1e-12)
    assert np.allclose(res["spacing y"], 0.0001616894788541336,
                       rtol=0, atol=1e-12)


def test_find_smooth_contour_abort():
    np.random.seed(47)
    x0 = np.random.normal(loc=100, scale=10, size=100)
    y0 = np.random.normal(loc=.1, scale=.01, size=100)

    ds = dclab.new_dataset({"area_um": x0, "deform": y0})

    event = threading.Event()
    event.set()

    res = smooth_contour.find_smooth_contour_spacing(
        ds_list=[ds],
        xax="area_um",
        yax="deform",
        xrange=(50, 150),
        yrange=(0.09, 0.11),
        quantiles=[0.90],
        abort_event=event,
    )

    assert res["total iterations"] == 0
    assert not res["success"]
    assert not res["corners found"]
    assert res["reason"] == "abort event"
    assert np.allclose(res["spacing x"], 1.1239763738160238,
                       rtol=0, atol=1e-12)
    assert np.allclose(res["spacing y"], 0.0005613691976395623,
                       rtol=0, atol=1e-12)


def test_find_smooth_contour_abort_during_iteration():
    """Aborting while iterating over datasets must not report success"""
    np.random.seed(47)
    x0 = np.random.normal(loc=100, scale=10, size=100)
    y0 = np.random.normal(loc=.1, scale=.01, size=100)

    ds = dclab.new_dataset({"area_um": x0, "deform": y0})

    class LateEvent(threading.Event):
        """Event that is set at the second call of `is_set`"""
        def __init__(self):
            super().__init__()
            self.calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls >= 2

    res = smooth_contour.find_smooth_contour_spacing(
        ds_list=[ds],
        xax="area_um",
        yax="deform",
        xrange=(50, 150),
        yrange=(0.09, 0.11),
        quantiles=[0.90],
        abort_event=LateEvent(),
    )

    # The abort event was set before the first contour was computed.
    assert res["total iterations"] == 1
    assert not res["success"]
    assert res["reason"] == "abort event"


@pytest.mark.parametrize("angles", [
    # NaN values and an opening angle of ~86 degrees
    [np.nan, np.nan, 1.5],
    # only NaN values (we cannot tell whether the contour is smooth)
    [np.nan, np.nan, np.nan],
])
def test_find_smooth_contour_spacing_nan_angles(angles, monkeypatch):
    """NaN opening angles must not result in "target opening angle reached"

    Before, `np.max` returned NaN and `max(0, nan)` returned 0.
    """
    monkeypatch.setattr(smooth_contour, "compute_contour_opening_angles",
                        lambda *args, **kwargs: np.array(angles))
    np.random.seed(47)
    x0 = np.random.normal(loc=100, scale=10, size=1000)
    y0 = np.random.normal(loc=.1, scale=.01, size=1000)
    ds = dclab.new_dataset({"area_um": x0, "deform": y0})

    res = smooth_contour.find_smooth_contour_spacing(
        ds_list=[ds],
        xax="area_um",
        yax="deform",
        xrange=(50, 150),
        yrange=(0.05, 0.15),
        quantiles=[0.5, 0.95],
    )
    assert res["reason"] != "target opening angle reached"
    assert res["total iterations"] > 1


def test_find_smooth_contour_spacing_all_contours(monkeypatch):
    """All contours of the highest percentile must be smooth (#288)"""
    phi = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([100 + 10 * np.cos(phi), .1 + .01 * np.sin(phi)],
                      axis=1)
    square = np.array([[60, .06], [70, .06], [70, .07], [60, .07],
                       [60, .06]])

    def get_contour_lines(self, *args, **kwargs):
        # the second contour (not smooth) must be taken into account
        return [[circle, square]]

    monkeypatch.setattr(dclab.kde.base.KernelDensityEstimator,
                        "get_contour_lines", get_contour_lines)
    np.random.seed(47)
    x0 = np.random.normal(loc=100, scale=10, size=1000)
    y0 = np.random.normal(loc=.1, scale=.01, size=1000)
    ds = dclab.new_dataset({"area_um": x0, "deform": y0})

    res = smooth_contour.find_smooth_contour_spacing(
        ds_list=[ds],
        xax="area_um",
        yax="deform",
        xrange=(50, 150),
        yrange=(0.05, 0.15),
        quantiles=[0.5, 0.95],
        max_iter=3,
    )
    assert not res["success"]
    assert res["reason"] == "maximum iterations reached"
    assert res["total iterations"] == 3
