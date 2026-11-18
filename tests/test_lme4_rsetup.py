"""Tests for `dclab.lme4.rsetup` that do not require R"""
import os
import sys

from dclab.lme4 import rsetup


def test_run_command_extends_environment(monkeypatch):
    """Passing `env` must extend the environment, not replace it"""
    monkeypatch.setenv("DCLAB_TEST_RSETUP", "peter")
    environ_before = dict(os.environ)
    script = ("import os; "
              "print(os.environ.get('DCLAB_TEST_RSETUP'), "
              "os.environ.get('R_LIBS_USER'))")
    out = rsetup.run_command(
        (sys.executable, "-c", script),
        env={"R_LIBS_USER": "/path/to/r/libs"},
    )
    assert out == "peter /path/to/r/libs"
    # the environment of the current process must not be modified
    assert dict(os.environ) == environ_before
