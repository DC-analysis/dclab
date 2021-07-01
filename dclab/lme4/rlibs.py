from distutils.version import LooseVersion
import importlib
import os

R_MIN_VERSION = "2.9.4"

R_SUBMODULES = [
    "rpy2.robjects.packages",
    "rpy2.situation",
    "rpy2.robjects.vectors",
]

R_SUBMODULES_3 = [
    "rpy2.rinterface_lib.callbacks",
]


class VersionError(BaseException):
    pass


class MockRPackage:
    def __getattr__(self, item):
        raise ImportError("Please install 'rpy2>={}'!".format(R_MIN_VERSION))


def import_r_submodules():
    importlib.import_module("rpy2.situation")
    r_home = rpy2.situation.get_r_home()
    if r_home is not None:
        if os.environ.get("R_HOME", None) is None:
            # set R_HOME globally (https://github.com/rpy2/rpy2/issues/796)
            os.environ["R_HOME"] = r_home
        if rpy2_is_version_3:
            mods = R_SUBMODULES + R_SUBMODULES_3
        else:
            mods = R_SUBMODULES
        for sm in mods:
            importlib.import_module(sm)


try:
    rpy2 = importlib.import_module("rpy2")
    if LooseVersion(rpy2.__version__) < LooseVersion(R_MIN_VERSION):
        raise VersionError("Please install 'rpy2>={}'!".format(R_MIN_VERSION))
except ImportError:
    rpy2 = MockRPackage()
    rpy2_is_version_3 = False
else:
    rpy2_is_version_3 = LooseVersion(rpy2.__version__) >= LooseVersion("3.0")
    import_r_submodules()
