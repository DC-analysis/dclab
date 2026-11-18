"""Tests for `dclab.lme4.wrapr` that do not require R"""
import os
import pathlib
import tempfile

import numpy as np
import pytest

import dclab
from dclab.lme4 import Rlme4, rsetup
from dclab.lme4.wrapr import arr2str


def test_arr2str():
    # integers must not be converted to floating point
    assert arr2str(np.array([1, 2, 2**53 + 1])) == "1,2,9007199254740993"
    assert arr2str(np.array(["c", "t"])) == "'c','t'"
    assert arr2str(np.array([1.5, 2])) == "1.5,2"


def test_fit_removes_temporary_script(monkeypatch):
    """The temporary R script must be closed and removed after a fit"""
    script_paths = []
    script_fds = []
    mkstemp = tempfile.mkstemp

    def mkstemp_wrapper(*args, **kwargs):
        fd, path = mkstemp(*args, **kwargs)
        script_fds.append(fd)
        return fd, path

    def fd_is_open_for(fd, path):
        """Whether `fd` is an open file descriptor of the file `path`"""
        try:
            fd_stat = os.fstat(fd)
        except OSError:
            return False  # closed
        # `fd` might have been reused for another file (other threads)
        return os.path.samestat(fd_stat, os.stat(path))

    def run_command(cmd, **kwargs):
        script_path = pathlib.Path(cmd[1])
        assert script_path.exists()
        # the file descriptor from `tempfile.mkstemp` must be closed
        assert not fd_is_open_for(script_fds[0], script_path)
        script_paths.append(script_path)
        return "Loading required package: lme4\nOUTPUT model: lmer #*#\n"

    monkeypatch.setattr(tempfile, "mkstemp", mkstemp_wrapper)
    monkeypatch.setattr(rsetup, "has_lme4", lambda: True)
    monkeypatch.setattr(rsetup, "get_r_script_path", lambda: "Rscript")
    monkeypatch.setattr(rsetup, "run_command", run_command)

    rlme4 = Rlme4(model="lmer", feature="deform")
    for group, repetition in [("control", 1), ("treatment", 1),
                              ("control", 2), ("treatment", 2)]:
        ds = dclab.new_dataset({"deform": [0.1, 0.2, 0.3]})
        ds.config["setup"]["chip region"] = "channel"
        rlme4.add_dataset(ds, group, repetition)

    res = rlme4.fit()
    assert res["model"] == "lmer"
    assert len(script_fds) == 1
    assert len(script_paths) == 1
    assert not script_paths[0].exists()


@pytest.mark.parametrize("bs_iter", [None, 1, 100])
def test_get_differential_dataset_bs_iter(monkeypatch, bs_iter):
    """The number of bootstrapping iterations can be set"""
    monkeypatch.setattr(rsetup, "has_lme4", lambda: True)
    kwargs = {} if bs_iter is None else {"bs_iter": bs_iter}
    rlme4 = Rlme4(model="glmer+loglink", feature="deform", **kwargs)
    for group, repetition in [("control", 1), ("treatment", 1)]:
        for region in ["channel", "reservoir"]:
            ds = dclab.new_dataset({"deform": [0.1, 0.2, 0.3, 0.4]})
            ds.config["setup"]["chip region"] = region
            rlme4.add_dataset(ds, group, repetition)
    features, groups, repetitions = rlme4.get_differential_dataset()
    assert groups == ["control", "treatment"]
    assert repetitions == [1, 1]
    expected = 1000 if bs_iter is None else bs_iter
    assert rlme4.bs_iter == expected
    assert [len(ff) for ff in features] == [expected, expected]


@pytest.mark.parametrize("bs_iter", [0, -1, 1.5, "10"])
def test_bs_iter_invalid(monkeypatch, bs_iter):
    monkeypatch.setattr(rsetup, "has_lme4", lambda: True)
    with pytest.raises(ValueError, match="must be a positive integer"):
        Rlme4(bs_iter=bs_iter)
