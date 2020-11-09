from distutils.version import LooseVersion
import importlib

R_MIN_VERSION = "2.9.4"

R_SUBMODULES = [
    "rpy2.robjects.packages",
    "rpy2.situation",
    "rpy2.robjects.vectors",
    "rpy2.rinterface_lib.callbacks",
    ]


class VersionError(BaseException):
    pass


class MockRPackage(object):
    def __getattr__(self, item):
        raise ImportError("Please install 'rpy2>={}'!".format(R_MIN_VERSION))


def import_r_submodules():
    importlib.import_module("rpy2.situation")
    if rpy2.situation.get_r_home() is None:
        return False
    else:
        for _sm in R_SUBMODULES:
            importlib.import_module(_sm)


try:
    rpy2 = importlib.import_module("rpy2")
    if LooseVersion(rpy2.__version__) < LooseVersion(R_MIN_VERSION):
        raise VersionError("Please install 'rpy2>={}'!".format(R_MIN_VERSION))
except ImportError:
    rpy2 = MockRPackage()
else:
    import_r_submodules()
