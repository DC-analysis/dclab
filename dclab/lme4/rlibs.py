import importlib
import os
import warnings

from ..external.packaging import parse as parse_version

#: Minimum R version
#: This is actually a dependency for rpy2, because the API changed then
#: (ffi.error: symbol 'R_tryCatchError' not found in library).
R_MIN_VERSION = "3.6.0"

#: Minimum rpy2 version
RPY2_MIN_VERSION = "2.9.4"

R_SUBMODULES = [
    "rpy2.robjects.packages",
    "rpy2.situation",
    "rpy2.robjects.vectors",
]

R_SUBMODULES_3 = [
    "rpy2.rinterface_lib.callbacks",
]


class RPY2UnavailableError(BaseException):
    pass


class RPY2ImportError(RPY2UnavailableError):
    pass


class RPY2OutdatedError(RPY2UnavailableError):
    pass


class RUnavailableError(BaseException):
    pass


class ROutdatedError(RUnavailableError):
    pass


class MockRPackage:
    def __init__(self, exception):
        self.exception = exception

    def __getattr__(self, item):
        raise self.exception


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
        try:
            for sm in mods:
                importlib.import_module(sm)
        except rpy2.rinterface_lib.openrlib.ffi.error as exc:
            # This error happens when the installed R version is too old:
            # "ffi.error: symbol 'R_tryCatchError' not found in library"
            raise ROutdatedError(
                f"Encountered '{exc.__class__.__name__}: {exc}'. "
                f"Please make sure you have 'R>={R_MIN_VERSION}'!")


try:
    rpy2 = importlib.import_module("rpy2")
    if parse_version(rpy2.__version__) < parse_version(RPY2_MIN_VERSION):
        raise RPY2OutdatedError(f"Please install 'rpy2>={RPY2_MIN_VERSION}'!")
except ImportError:
    rpy2 = MockRPackage(
        RPY2ImportError(f"Please install 'rpy2>={RPY2_MIN_VERSION}'!"))
    rpy2_is_version_3 = False
except BaseException as e:
    rpy2 = MockRPackage(e)
    rpy2_is_version_3 = False
else:
    rpy2_is_version_3 = parse_version(rpy2.__version__) >= parse_version("3.0")
    try:
        import_r_submodules()
    except RUnavailableError as e:
        warnings.warn("There is an issue with the linked R version: "
                      + f"{e.__class__.__name__}: {e}")
