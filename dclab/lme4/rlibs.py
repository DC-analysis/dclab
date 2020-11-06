from distutils.version import LooseVersion
import importlib

LIB_MIN_VERSIONS = {
    "rpy2": "3.3.0",
}

LIB_SUBMODULES = {
    "rpy2": ["rpy2.robjects.packages",
             "rpy2.situation",
             "rpy2.robjects.vectors",
             "rpy2.rinterface_lib.callbacks"
             ]
}


class MockPackage(object):
    def __init__(self, name):
        self.name = name

    def __getattr__(self, item):
        raise ImportError("Please install '{}>={}'!".format(
            self.name, LIB_MIN_VERSIONS[self.name]))


for _lib in LIB_MIN_VERSIONS:
    _mods = {}
    try:
        _mods[_lib] = importlib.import_module(_lib)
        _v_req = LIB_MIN_VERSIONS[_lib]
        if LooseVersion(_mods[_lib].__version__) < LooseVersion(_v_req):
            raise ValueError("Please install '{}>={}'!".format(_lib, _v_req))
        # install submodules
        if _lib in LIB_SUBMODULES:
            for _sm in LIB_SUBMODULES[_lib]:
                _mods[_sm] = importlib.import_module(_sm)
    except (ImportError, ValueError):
        _mods[_lib] = MockPackage(_lib)
    locals().update(_mods)
