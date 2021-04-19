"""Utility methods"""
import hashlib
import pathlib
import warnings

import h5py
import numpy as np


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
    hasher = constructor()
    fname = pathlib.Path(fname)
    with fname.open('rb') as fd:
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
    """Bytes representation of an object for hashing"""
    if isinstance(obj, str):
        return obj.encode("utf-8")
    elif isinstance(obj, pathlib.Path):
        return obj2bytes(str(obj))
    elif isinstance(obj, (bool, int, float)):
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
        return obj2bytes(obj.identifier)
    elif isinstance(obj, h5py.Dataset):
        return obj2bytes(obj[0])
    else:
        raise ValueError("No rule to convert object '{}' to string.".
                         format(obj.__class__))
