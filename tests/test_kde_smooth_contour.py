import threading

import numpy as np

import dclab
from dclab.kde import smooth_contour


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
    angles = smooth_contour.compute_contour_opening_angles(
        contour=contour,
        xrange=[0, 1],
        yrange=[0, 1],
        xscale="log",
        yscale="log",
    )
    assert np.allclose(angles, np.pi/3)


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

    assert res['total iterations'] == 3
    assert res['success']
    assert not res["corners found"]
    assert res['reason'] == 'maximum contour length reached'
    assert np.allclose(res['spacing x'], 0.28099409345400594,
                       rtol=0, atol=1e-12)
    assert np.allclose(res['spacing y'], 0.00014034229940989058,
                       rtol=0, atol=1e-12)


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

    assert res['total iterations'] == 1
    assert not res['success']
    assert not res["corners found"]
    assert res['reason'] == 'maximum iterations reached'
    assert np.allclose(res['spacing x'], 0.46201586550424045,
                       rtol=0, atol=1e-12)
    assert np.allclose(res['spacing y'], 0.00024627719927894213,
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

    assert res['total iterations'] == 3
    assert res['success']
    assert not res["corners found"]
    assert res['reason'] == 'target opening angle reached'
    assert np.allclose(res['spacing x'], 0.27646847583098244,
                       rtol=0, atol=1e-12)
    assert np.allclose(res['spacing y'], 0.00016320865830594234,
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

    assert res['total iterations'] == 3
    assert res['success']
    assert res['reason'] == 'maximum contour length reached'
    assert res["corners found"]
    assert np.allclose(res['spacing x'], 0.16515957250959012,
                       rtol=0, atol=1e-12)
    assert np.allclose(res['spacing y'], 0.0001616894788541336,
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

    assert res['total iterations'] == 0
    assert not res['success']
    assert not res["corners found"]
    assert res['reason'] == 'abort event'
    assert np.allclose(res['spacing x'], 1.1239763738160238,
                       rtol=0, atol=1e-12)
    assert np.allclose(res['spacing y'], 0.0005613691976395623,
                       rtol=0, atol=1e-12)
