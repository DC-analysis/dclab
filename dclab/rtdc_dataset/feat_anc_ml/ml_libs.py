import importlib

from ...external.packaging import parse as parse_version


class MockPackage:
    def __init__(self, name, min_version):
        self.name = name
        self.min_version = min_version

    def __getattr__(self, item):
        raise ImportError(f"Please install '{self.name}>={self.min_version}'!")


def import_or_mock_package(name, min_version):
    if name not in _locals:
        try:
            _mod = importlib.import_module(name)
            if parse_version(_mod.__version__) < parse_version(min_version):
                raise ValueError(f"Please install '{name}>={min_version}'!")
        except ImportError:
            _mod = MockPackage(name, min_version)
    else:
        _mod = _locals[name]
    _locals[name] = _mod
    return _mod


_locals = locals()
