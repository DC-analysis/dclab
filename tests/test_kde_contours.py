#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

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

    if __name__ == "__main__":
        import matplotlib.pylab as plt
        plt.plot(ds["area_um"], ds["deform"], "x")
        # There should be 11 points (q=.89) in the contour
        for cc in contours:
            plt.plot(cc[:, 0], cc[:, 1])
        plt.show()

    nump = 0
    for p in zip(x0, y0):
        nump += polygon_filter.PolygonFilter.point_in_poly(p, poly=contours[0])

    assert nump == 11, "there should be (1-q)*100 points in the contour"

    # added in dclab 0.24.1
    nump2 = skimage.measure.points_in_poly(
        np.concatenate((x0.reshape(-1, 1), y0.reshape(-1, 1)), axis=1),
        contours[0])
    assert nump2.sum() == 11


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

    if __name__ == "__main__":
        c1 = kde_contours.find_contours_level(density=kde,
                                              x=x,
                                              y=y,
                                              level=level,
                                              closed=True)[0]
        c2 = kde_contours.find_contours_level(density=kde,
                                              x=x,
                                              y=y,
                                              level=level2,
                                              closed=True)[0]
        import matplotlib.pylab as plt
        plt.plot(ds["area_um"], ds["deform"], "x")
        # The contours are right on top of each other
        plt.plot(c1[:, 0], c2[:, 1])
        plt.plot(c1[:, 0], c2[:, 1], ls="--")
        plt.show()


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
