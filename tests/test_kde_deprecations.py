import numpy as np
import pytest

import dclab


def test_kde_methods_deprecated_warning():
    with pytest.deprecated_call():
        a = np.arange(100)
        b = dclab.kde_methods.bin_width_doane(a)
        assert np.allclose(b, 12.951578044133464)
