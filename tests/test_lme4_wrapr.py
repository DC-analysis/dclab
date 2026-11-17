"""Tests for `dclab.lme4.wrapr` that do not require R"""
import numpy as np

from dclab.lme4.wrapr import arr2str


def test_arr2str():
    # integers must not be converted to floating point
    assert arr2str(np.array([1, 2, 2**53 + 1])) == "1,2,9007199254740993"
    assert arr2str(np.array(["c", "t"])) == "'c','t'"
    assert arr2str(np.array([1.5, 2])) == "1.5,2"
