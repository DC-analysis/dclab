import numpy as np
import dclab.kde.binning


def test_bin_width_doane():
    a = np.arange(100)
    b = dclab.kde.binning.bin_width_doane(a)
    assert np.allclose(b, 12.951578044133464)


def test_bin_width_percentile():
    a = np.arange(100)
    b = dclab.kde.binning.bin_width_percentile(a)
    assert np.allclose(b, 3.4434782608695653)


def test_bin_width_percentile_out_of_range():
    """percentiles might be out of range"""
    a = np.zeros(1000)
    a[0] = 1
    a[1] = 2
    b = dclab.kde.binning.bin_width_percentile(a)
    assert b != 0
    assert np.allclose(b, 0.2)


def test_bin_width_percentile_zero():
    """percentiles might be out of range"""
    a = np.zeros(1000)
    b = dclab.kde.binning.bin_width_percentile(a)
    assert b != 0
    assert np.allclose(b, 1)
