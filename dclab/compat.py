# python 2+3 compatibility functions
from __future__ import unicode_literals
import io
import sys


if sys.version_info[0] == 2:
    from backports.functools_lru_cache import lru_cache  # noqa: F401
    pyver = 2
    str_types = basestring  # noqa: F821
    hdf5_str = unicode  # noqa: F821
    PyImportError = ImportError

    def is_file_obj(obj):
        return isinstance(obj, file)  # noqa: F821
    def to_unicode(obj):
        return unicode(obj)
else:
    pyver = 3
    from functools import lru_cache  # noqa: F401
    if sys.version_info[1] <= 5:
        PyImportError = ImportError
    else:
        PyImportError = ModuleNotFoundError  # noqa: F821
    str_types = str
    hdf5_str = str

    def is_file_obj(obj):
        return isinstance(obj, io.IOBase)
    def to_unicode(obj):
        return str(obj)