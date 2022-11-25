"""Utility methods"""
import functools
import hashlib
import numbers
import pathlib
import warnings

import h5py
import numpy as np
from .rtdc_dataset.config import Configuration, ConfigurationDict


class file_monitoring_lru_cache:
    """Decorator for caching data extracted from files

    The function that is decorated with `file_monitoring_lru_cache`
    must accept `path` as its first argument. Caching is
    done with an `lru_cache`. In addition to the full path
    and the other arguments to the decorated function, the
    size and the modification time of `path` is used as a
    key for the decorator.
    If the path does not exist, no caching is done.

    Use case: Extract and cache metadata from a file on disk
    that may change.
    """
    def __init__(self, maxsize=100):
        self.lru_cache = functools.lru_cache(maxsize=maxsize)
        self.cached_wrapper = None

    def __call__(self, func):
        @self.lru_cache
        def cached_wrapper(path, path_stats, *args, **kwargs):
            assert path_stats, "We need stat for validating the cache"
            return func(path, *args, **kwargs)

        @functools.wraps(func)
        def wrapper(path, *args, **kwargs):
            full_path = pathlib.Path(path).resolve()
            if full_path.exists():
                path_stat = full_path.stat()
                return cached_wrapper(
                    path=full_path,
                    path_stats=(path_stat.st_mtime_ns, path_stat.st_size),
                    *args,
                    **kwargs)
            else:
                # `func` will most-likely raise an exception
                return func(path, *args, **kwargs)

        wrapper.cache_clear = cached_wrapper.cache_clear
        wrapper.cache_info = cached_wrapper.cache_info

        return wrapper


@file_monitoring_lru_cache(maxsize=100)
def hashfile(fname, blocksize=65536, count=0, constructor=hashlib.md5,
             hasher_class=None):
    """Compute md5 hex-hash of a file

    Parameters
    ----------
    fname: str or pathlib.Path
        path to the file
    blocksize: int
        block size in bytes read from the file
        (set to `0` to hash the entire file)
    count: int
        number of blocks read from the file
    hasher_class: callable
        deprecated, see use `constructor` instead
    constructor: callable
        hash algorithm constructor
    """
    if hasher_class is not None:
        warnings.warn("The `hasher_class` argument is deprecated, please use "
                      "`constructor` instead.")
        constructor = hasher_class

    path = pathlib.Path(fname)

    hasher = constructor()
    with path.open('rb') as fd:
        buf = fd.read(blocksize)
        ii = 0
        while len(buf) > 0:
            hasher.update(buf)
            buf = fd.read(blocksize)
            ii += 1
            if count and ii == count:
                break
    return hasher.hexdigest()


def hashobj(obj):
    """Compute md5 hex-hash of a Python object"""
    return hashlib.md5(obj2bytes(obj)).hexdigest()


def obj2bytes(obj):
    """Bytes representation of an object for hashing

    Note that there is no guarantee that the bytes representation
    returned is reproducible across sessions. This is currently the
    case when an :class:`.RTDCBase` instance is passed. There is no
    opinion on wether/how this should be changed.
    """
    if isinstance(obj, str):
        return obj.encode("utf-8")
    elif isinstance(obj, pathlib.Path):
        return obj2bytes(str(obj))
    elif isinstance(obj, (bool, numbers.Number)):
        return str(obj).encode("utf-8")
    elif obj is None:
        return b"none"
    elif isinstance(obj, np.ndarray):
        return obj.tobytes()
    elif isinstance(obj, tuple):
        return obj2bytes(list(obj))
    elif isinstance(obj, list):
        return b"".join(obj2bytes(o) for o in obj)
    elif isinstance(obj, dict):
        return obj2bytes(sorted(obj.items()))
    elif hasattr(obj, "identifier"):
        # For RTDCBase, this identifier is not reproducible in-between
        # sessions. We might want to change this to something that is
        # reproducible in the future (if the need arises).
        return obj2bytes(obj.identifier)
    elif isinstance(obj, h5py.Dataset):
        return obj2bytes([
            # path in the HDF5 file
            obj.name,
            # filename
            obj.file.filename,
            # when the file was changed
            pathlib.Path(obj.file.filename).stat().st_mtime,
            # size of the file
            pathlib.Path(obj.file.filename).stat().st_size,
            ])
    elif isinstance(obj, Configuration):
        return obj2bytes(obj.tostring())
    elif isinstance(obj, ConfigurationDict):
        return obj2bytes(dict(obj))
    else:
        raise ValueError("No rule to convert object '{}' to string.".
                         format(obj.__class__))
