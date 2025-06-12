import numpy as np
import pytest
import dclab
from dclab.external import skimage
from dclab.kde import KernelDensityEstimator
from dclab import polygon_filter

from helper_methods import example_data_dict


def test_contour_lines():
    np.random.seed(47)
    x0 = np.random.normal(loc=100, scale=10, size=100)
    y0 = np.random.normal(loc=.1, scale=.01, size=100)

    ds = dclab.new_dataset({"area_um": x0, "deform": y0})
    ds.config["filtering"]["enable filters"] = False

    kde_instance = KernelDensityEstimator(ds)

    contours = kde_instance.get_contour_lines(xax="area_um",
                                              yax="deform",
                                              xacc=.10,
                                              yacc=.01,
                                              kde_type="histogram",
                                              kde_kwargs=None,
                                              xscale="linear",
                                              yscale="linear",
                                              quantiles=[0.89])
    nump = 0
    for p in zip(x0, y0):
        nump += polygon_filter.PolygonFilter.point_in_poly(p,
                                                           poly=contours[0][0])

    assert nump == 11, "there should be (1-q)*100 points in the contour"

    # added in dclab 0.24.1
    nump2 = skimage.measure.points_in_poly(
        np.concatenate((x0.reshape(-1, 1), y0.reshape(-1, 1)), axis=1),
        contours[0][0])
    assert nump2.sum() == 11


def test_contour_lines_without_quantiles_with_ret_levels():
    np.random.seed(47)
    x0 = np.random.normal(loc=100, scale=10, size=100)
    y0 = np.random.normal(loc=.1, scale=.01, size=100)

    ds = dclab.new_dataset({"area_um": x0, "deform": y0})
    ds.config["filtering"]["enable filters"] = False

    kde_instance = KernelDensityEstimator(ds)

    contours, levels = kde_instance.get_contour_lines(xax="area_um",
                                                      yax="deform",
                                                      xacc=.10,
                                                      yacc=.01,
                                                      kde_type="histogram",
                                                      kde_kwargs=None,
                                                      xscale="linear",
                                                      yscale="linear",
                                                      ret_levels=True,)
    nump = 0
    for p in zip(x0, y0):
        nump += polygon_filter.PolygonFilter.point_in_poly(p,
                                                           poly=contours[0][1])

    assert nump == 50, "there should be (1-q)*100 points in the contour"

    # added in dclab 0.24.1
    nump2 = skimage.measure.points_in_poly(
        np.concatenate((x0.reshape(-1, 1), y0.reshape(-1, 1)), axis=1),
        contours[0][1])
    assert nump2.sum() == 50

    assert len(levels) == 2, "there should be two levels"


@pytest.mark.filterwarnings("ignore::UserWarning", "ignore::RuntimeWarning")
def test_contour_lines_data_with_too_much_space():
    np.random.seed(47)
    x0 = np.random.normal(loc=100, scale=10, size=100)
    y0 = np.random.normal(loc=.1, scale=.01, size=100)

    n = 100  # Adjust this to control spacing
    x0 = x0[::n]
    y0 = y0[::n]

    ds = dclab.new_dataset({"area_um": x0, "deform": y0})
    ds.config["filtering"]["enable filters"] = False

    kde_instance = KernelDensityEstimator(ds)

    contours = kde_instance.get_contour_lines(xax="area_um",
                                              yax="deform",
                                              xacc=.10,
                                              yacc=.01,
                                              kde_type="histogram",
                                              kde_kwargs=None,
                                              xscale="linear",
                                              yscale="linear",
                                              quantiles=[0.5])
    assert not contours, "there should be no contours with too much space"


def test_kde_log_get_at():
    ddict = example_data_dict(size=300, keys=["area_um", "deform"])
    ddict["deform"][:20] = .1
    ddict["area_um"][:20] = .5
    ds = dclab.new_dataset(ddict)
    kde_instance = KernelDensityEstimator(ds)
    a = kde_instance.get_at(xax="area_um", yax="deform", yscale="log")
    assert np.all(a[:20] == a[0])


def test_kde_log_get_at_points():
    ddict = example_data_dict(size=300, keys=["area_um", "tilt"])
    ds = dclab.new_dataset(ddict)
    kde_instance = KernelDensityEstimator(ds)
    a = kde_instance.get_at(yscale="log", xax="area_um", yax="tilt")
    b = kde_instance.get_at(yscale="log", xax="area_um", yax="tilt")

    assert np.all(a == b)


def test_kde_log_get_at_invalid():
    ddict = example_data_dict(size=300, keys=["area_um", "deform"])
    ddict["deform"][:20] = .1
    ddict["area_um"][:20] = .5
    ddict["deform"][21] = np.nan
    ddict["deform"][22] = np.inf
    ddict["deform"][23] = -.1
    ds = dclab.new_dataset(ddict)
    kde_instance = KernelDensityEstimator(ds)
    a = kde_instance.get_at(xax="area_um", yax="deform", yscale="log")
    assert np.all(a[:20] == a[0])
    assert np.isnan(a[21])
    assert np.isnan(a[22])
    assert np.isnan(a[23])


def test_kde_get_at_positions():
    ddict = example_data_dict()
    ds = dclab.new_dataset(ddict)

    kde_instance = KernelDensityEstimator(ds)

    ds.config["filtering"]["enable filters"] = False
    sc = kde_instance.get_at(xax="area_um", yax="deform")
    sc2 = kde_instance.get_at(xax="area_um", yax="deform",
                              positions=(ds["area_um"], ds["deform"]))
    assert np.all(sc == sc2)


def test_kde_log_get_at_out_of_bounds():
    ddict = example_data_dict(size=300, keys=["area_um", "deform"])
    ds = dclab.new_dataset(ddict)
    kde_instance = KernelDensityEstimator(ds)

    # Define a positions that has values outside the typical data range
    # `area_um` ([0.0, 400]) and `deform` ([0.0, 0.02])
    positions = ([410, 300, 300, 300],
                 [-1, -2, 0.01, 0.015])

    # Get the density at the out-of-bounds position
    a = kde_instance.get_at(xax="area_um", yax="deform",
                            positions=positions, yscale="log")
    assert np.isnan(a[0])
    assert np.isnan(a[1])
    assert np.isfinite(a[2])
    assert np.isfinite(a[3])
