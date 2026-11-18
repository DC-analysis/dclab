"""Tests for `dclab.lme4.wrapr` that do not require R"""
import os
import pathlib
import tempfile

import numpy as np

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
