"""
Testing script for volume.py
An ellipse is created and the analytical and numerical solution are compared
"""
import itertools

import numpy as np
import pytest

import dclab
from dclab.features.volume import get_volume, vol_revolve

from helper_methods import retrieve_data


def area_of_polygon(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def centroid_of_polygon(points):
    """
    http://stackoverflow.com/a/14115494/190597 (mgamba)
    Centroid of polygon:
    http://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon
    """
    area = area_of_polygon(*zip(*points))
    result_x = 0
    result_y = 0
    N = len(points)
    points = itertools.cycle(points)
    x1, y1 = next(points)
    for _i in range(N):
        x0, y0 = x1, y1
        x1, y1 = next(points)
        cross = (x0 * y1) - (x1 * y0)
        result_x += (x0 + x1) * cross
        result_y += (y0 + y1) * cross
    result_x /= (area * 6.0)
    result_y /= (area * 6.0)
    return abs(result_x), abs(result_y)


def get_ellipse_coords(a, b, x=0.0, y=0.0, angle=0.0, k=2):
    """

    Draws an ellipse using (360*k + 1) discrete points; based on pseudo code
    given at http://en.wikipedia.org/wiki/Ellipse

    k = 1 means 361 points (degree by degree)
    a = major axis distance,
    b = minor axis distance,
    x = offset along the x-axis
    y = offset along the y-axis
    angle = clockwise rotation [in degrees] of the ellipse;
        * angle=0  : the ellipse is aligned with the positive x-axis
        * angle=30 : rotated 30 degrees clockwise from positive x-axis
    """
    pts = np.zeros((360 * k + 1, 2))

    beta = -angle * np.pi / 180
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    alpha = np.linspace(0, 2*np.pi, 360 * k + 1, endpoint=True)

    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)

    pts[:, 0] = x + (a * cos_alpha * cos_beta - b * sin_alpha * sin_beta)
    pts[:, 1] = y + (a * cos_alpha * sin_beta + b * sin_alpha * cos_beta)

    return pts


def test_af_volume():
    pytest.importorskip("nptdms")
    ds = dclab.new_dataset(retrieve_data("fmt-tdms_minimal_2016.zip"))
    vol = ds["volume"]
    # There are a lot of nans, because the contour is not given everywhere
    vol = vol[~np.isnan(vol)]
    assert np.allclose(vol[0], 571.0512910465295)
    assert np.allclose(vol[12], 1008.5405681260276)


def test_get_volume():
    # Helper definitions to get an ellipse
    major = 10
    minor = 5
    ellip = get_ellipse_coords(a=major,
                               b=minor,
                               x=minor,
                               y=5,
                               angle=0,
                               k=100)
    # obtain the centroid (corresponds to pos_x and pos_lat)
    cx, cy = centroid_of_polygon(ellip)
    volume = get_volume(cont=[ellip],
                        pos_x=[cx],
                        pos_y=[cy],
                        pix=1,
                        fix_orientation=True)

    # Analytic solution for volume of ellipsoid:
    v = 4 / 3 * np.pi * major * minor**2
    msg = "Calculation of volume is faulty!"
    assert np.allclose(np.array(volume), np.array(v)), msg


def test_orientation():
    """counter-clockwise vs clockwise"""
    # Helper definitions to get an ellipse
    major = 10
    minor = 5
    ellip = get_ellipse_coords(a=major,
                               b=minor,
                               x=minor,
                               y=5,
                               angle=0,
                               k=100)
    # obtain the centroid (corresponds to pos_x and pos_lat)
    cx, cy = centroid_of_polygon(ellip)

    v1 = get_volume(cont=ellip,
                    pos_x=cx,
                    pos_y=cy,
                    pix=1,
                    fix_orientation=True)

    # Turn contour around
    v2 = get_volume(cont=ellip[::-1, :],
                    pos_x=cx,
                    pos_y=cy,
                    pix=1,
                    fix_orientation=True)

    assert np.allclose(v1, v2, atol=1e-11, rtol=0)


def test_shape():
    major = 10
    minor = 5
    ellip = get_ellipse_coords(a=major,
                               b=minor,
                               x=minor,
                               y=5.0,
                               angle=0,
                               k=100)
    cx, cy = centroid_of_polygon(ellip)
    # no lists
    volume = get_volume(cont=ellip,
                        pos_x=cx,
                        pos_y=cy,
                        pix=1)
    assert isinstance(volume, float)

    volumelist = get_volume(cont=[ellip],
                            pos_x=[cx],
                            pos_y=[cy],
                            pix=1)
    assert isinstance(volumelist, np.ndarray)


def test_xpos():
    """xpos is not necessary to compute volume dense ellipse"""
    major = 10
    minor = 5
    ellip = get_ellipse_coords(a=major,
                               b=minor,
                               x=minor,
                               y=5.0,
                               angle=0,
                               k=100)
    cx, cy = centroid_of_polygon(ellip)
    # no lists
    v0 = get_volume(cont=ellip,
                    pos_x=cx,
                    pos_y=cy,
                    pix=1)
    for cxi in np.linspace(0, 2 * cx, 10):
        vi = get_volume(cont=ellip,
                        pos_x=cxi,
                        pos_y=cy,
                        pix=1)
        assert np.allclose(v0, vi)


@pytest.mark.parametrize("npoints,rtol", [[100, 6.72e-4],
                                          [1000, 6.6e-6]])
def test_vol_revolve_circular_toroid(npoints, rtol):
    """Upstream test function 1

    https://de.mathworks.com/matlabcentral/fileexchange/36525-volrevolve

    % Verification for circular toroid of major radius R0, minor radius a
    % Volume is 2 * pi^2 * R0 * a^2. Run this code:
    % clear all
    % R0 = 5 ;
    % a = 1 ;
    % npoints = 100 ;
    % theta = 2*pi*[0:1:npoints-1]'/double(npoints-1) ;
    % R = R0 + a*cos(theta) ;
    % Z =      a*sin(theta) ;
    % vol_analytic = 2 * pi^2 * R0 * a^2 ;
    %  >> 98.6960
    % vol = volRevolve(R,Z) ;
    %  >> 98.6298 (6.7e-04 relative error)
    % Do it again with npoints = 1000, get:
    %  >> 98.6954 (6.6e-06 relative error)
    % As expected, it's always slightly small because the polygon inscribes the
    % circle.
    """
    r0 = 5
    a = 1
    theta = 2 * np.pi * np.arange(npoints-1) / (npoints-1)
    r = r0 + a*np.cos(theta)
    z = a*np.sin(theta)
    vol_analytic = 2 * np.pi**2 * r0 * a**2
    vol = vol_revolve(r, z)
    assert np.allclose(vol_analytic, 98.6960, rtol=0, atol=0.001)
    assert np.allclose(vol_analytic, vol, rtol=rtol, atol=0)
    assert vol < vol_analytic


def test_vol_revolve_rectangular_toroid():
    """Upstream test function 2

    https://de.mathworks.com/matlabcentral/fileexchange/36525-volrevolve

    % Verification for washer (rectangular toroid), with the radius of the
    % 'hole' in the washer being a, and the outer radius of the washer being b.
    % (Thus the width of the metal cross section is b-a.) The height of the
    % washer is h. Then the volume is pi * (b^2 - a^2) * h. Run this code:
    clear all
    a = 1 ;
    b = 2 ;
    h = 10 ;
    R = [a; b; b; a; a] ;
    Z = [0; 0; h; h; 0] ;
    vol_analytic = pi * (b^2 - a^2) * h ;
    % >> 94.2478
    vol = volRevolve(R,Z) ;
    % >> 94.2478
    """
    a = 1
    b = 2
    h = 10
    r = [a, b, b, a, a]
    z = [0, 0, h, h, 0]
    vol_analytic = np.pi * (b**2 - a**2) * h
    vol = vol_revolve(r, z)
    assert vol_analytic == vol
    assert np.allclose(vol, 94.2478, rtol=0, atol=0.0001)
