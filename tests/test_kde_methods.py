import numpy as np
import dclab.kde.methods


def test_bin_width_doane():
    a = np.arange(100)
    b = dclab.kde.methods.bin_width_doane(a)
    assert np.allclose(b, 12.951578044133464)


def test_bin_width_percentile():
    a = np.arange(100)
    b = dclab.kde.methods.bin_width_percentile(a)
    assert np.allclose(b, 3.4434782608695653)
