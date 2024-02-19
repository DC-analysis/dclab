import numpy as np
import dclab
from dclab.external import skimage
from dclab import kde_contours
from dclab import polygon_filter


def test_contour_basic():
    np.random.seed(47)
    x0 = np.random.normal(loc=100, scale=10, size=100)
    y0 = np.random.normal(loc=.1, scale=.01, size=100)

    ds = dclab.new_dataset({"area_um": x0, "deform": y0})
    ds.config["filtering"]["enable filters"] = False

    x, y, kde = ds.get_kde_contour(xax="area_um",
                                   yax="deform",
                                   xacc=.10,
                                   yacc=.01,
                                   kde_type="histogram")
    level = kde_contours.get_quantile_levels(density=kde,
                                             x=x,
                                             y=y,
                                             xp=ds["area_um"],
                                             yp=ds["deform"],
                                             q=.89,
                                             normalize=True)
    contours = kde_contours.find_contours_level(density=kde,
                                                x=x,
                                                y=y,
                                                level=level,
                                                closed=True)

    nump = 0
    for p in zip(x0, y0):
        nump += polygon_filter.PolygonFilter.point_in_poly(p, poly=contours[0])

    assert nump == 11, "there should be (1-q)*100 points in the contour"

    # added in dclab 0.24.1
    nump2 = skimage.measure.points_in_poly(
        np.concatenate((x0.reshape(-1, 1), y0.reshape(-1, 1)), axis=1),
        contours[0])
    assert nump2.sum() == 11


def test_contour_user_put_zero_accuracy():
    np.random.seed(47)
    x0 = np.random.normal(loc=100, scale=10, size=100)
    y0 = np.random.normal(loc=.1, scale=.01, size=100)

    ds = dclab.new_dataset({"area_um": x0, "deform": y0})
    ds.config["filtering"]["enable filters"] = False

    x, y, kde = ds.get_kde_contour(xax="area_um",
                                   yax="deform",
                                   xacc=0,  # testing zero-valued accuracy
                                   yacc=.01,
                                   kde_type="histogram")
    assert np.allclose(x[0][0], 74.24317287410939, atol=1e-12, rtol=0)
    assert np.allclose(y[0][0], 0.07748466975161497, atol=1e-12, rtol=0)
    assert np.allclose(kde[0][0], 0, atol=1e-12, rtol=0)


def test_percentile():
    np.random.seed(47)
    x0 = np.random.normal(loc=100, scale=10, size=1000)
    y0 = np.random.normal(loc=.1, scale=.01, size=1000)

    ds = dclab.new_dataset({"area_um": x0, "deform": y0})
    ds.config["filtering"]["enable filters"] = False

    x, y, kde = ds.get_kde_contour(xax="area_um",
                                   yax="deform",
                                   xacc=.10,
                                   yacc=.01,
                                   kde_type="histogram")
    level = kde_contours.get_quantile_levels(density=kde,
                                             x=x,
                                             y=y,
                                             xp=ds["area_um"],
                                             yp=ds["deform"],
                                             q=.89,
                                             normalize=True)

    level2, err = kde_contours._find_quantile_level(density=kde,
                                                    x=x,
                                                    y=y,
                                                    xp=ds["area_um"],
                                                    yp=ds["deform"],
                                                    quantile=.89,
                                                    acc=0,
                                                    ret_err=True)
    # since _find_quantile level does not do linear interpolation
    # in the density, the computed values can differ from the values
    # obtained using get_quantile_levels - even with err==0.
    assert err == 0
    # This is the resulting level difference.
    assert np.abs(level - level2) < 0.00116
