import atexit
import shutil
import tempfile
import time

TMPDIR = tempfile.mkdtemp(prefix=time.strftime(
    "dclab_test_%H.%M_"))


def pytest_configure(config):
    """
    Allows plugins and conftest files to perform initial configuration.
    This hook is called for every plugin and initial conftest
    file after command line options have been parsed.
    """
    tempfile.tempdir = TMPDIR
    atexit.register(shutil.rmtree, TMPDIR, ignore_errors=True)
