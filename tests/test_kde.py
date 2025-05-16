import numpy as np
import pytest
import dclab
from dclab.external import skimage
from dclab.kde import KernelDensityEstimator
from dclab import polygon_filter


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
