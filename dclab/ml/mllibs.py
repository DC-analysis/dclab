from distutils.version import LooseVersion
import importlib

LIB_MIN_VERSIONS = {
    "tensorflow": "2.0",
}


class MockPackage(object):
    def __init__(self, name):
        self.name = name

    def __getattr__(self, item):
        raise ImportError("Please install '{}>={}'!".format(
            self.name, LIB_MIN_VERSIONS[self.name]))


for _lib in LIB_MIN_VERSIONS:
    try:
        _mod = importlib.import_module(_lib)
        _v_req = LIB_MIN_VERSIONS[_lib]
        if LooseVersion(_mod.__version__) < LooseVersion(_v_req):
            raise ValueError("Please install '{}>={}'!".format(_lib, _v_req))
    except ImportError:
        _mod = MockPackage(_lib)
    locals()[_lib] = _mod
