
import numpy as np
import dclab


def test_bin_width_doane():
    a = np.arange(100)
    b = dclab.kde_methods.bin_width_doane(a)
    assert np.allclose(b, 12.951578044133464)


def test_bin_width_percentile():
    a = np.arange(100)
    b = dclab.kde_methods.bin_width_percentile(a)
    assert np.allclose(b, 3.4434782608695653)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
