import pytest
import numpy as np
import dclab


def test_kde_contours_deprecated_warning():
    with pytest.deprecated_call():
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
        level = dclab.kde_contours.get_quantile_levels(
            density=kde,
            x=x,
            y=y,
            xp=ds["area_um"],
            yp=ds["deform"],
            q=.89,
            normalize=True
        )

        level2, err = dclab.kde_contours._find_quantile_level(
            density=kde,
            x=x,
            y=y,
            xp=ds["area_um"],
            yp=ds["deform"],
            quantile=.89,
            acc=0,
            ret_err=True
        )
        # since _find_quantile level does not do linear interpolation
        # in the density, the computed values can differ from the values
        # obtained using get_quantile_levels - even with err==0.
        assert err == 0
        # This is the resulting level difference.
        assert np.abs(level - level2) < 0.00116


def test_kde_methods_deprecated_warning():
    with pytest.deprecated_call():
        a = np.arange(100)
        b = dclab.kde_methods.bin_width_doane(a)
        assert np.allclose(b, 12.951578044133464)
