import importlib
import sys

import numpy as np
import pytest

import dclab


def test_kde_methods_deprecated_warning():
    with pytest.deprecated_call():
        a = np.arange(100)
        b = dclab.kde_methods.bin_width_doane(a)
        assert np.allclose(b, 12.951578044133464)


def test_kde_contours_deprecated_warning():
    # make sure the module is imported (and the warning is issued) here
    sys.modules.pop("dclab.kde_contours", None)
    with pytest.deprecated_call():
        kde_contours = importlib.import_module("dclab.kde_contours")
    assert (kde_contours.get_quantile_levels
            is dclab.kde.contours.get_quantile_levels)
    assert (kde_contours.find_contours_level
            is dclab.kde.contours.find_contours_level)
