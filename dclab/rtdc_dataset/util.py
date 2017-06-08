#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Utility methods"""
from __future__ import division, print_function, unicode_literals

import hashlib
import sys

import numpy as np

if sys.version_info[0] == 2:
    str_classes = (str, unicode)
else:
    str_classes = str
    

def hashfile(fname, blocksize=65536):
    afile = open(fname, 'rb')
    hasher = hashlib.sha256()
    buf = afile.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read(blocksize)
    afile.close()
    return hasher.hexdigest()


def obj2str(obj):
    """Full string representation of an object for hashing"""
    if isinstance(obj, str_classes):
        return obj.encode("utf-8")
    elif isinstance(obj, (bool, int, float)):
        return str(obj).encode("utf-8")
    elif obj is None:
        return b"none"
    elif isinstance(obj, np.ndarray):
        return obj.tostring()
    elif isinstance(obj, tuple):
        return obj2str(list(obj))
    elif isinstance(obj, list):
        return b"".join(obj2str(o) for o in obj)
    elif isinstance(obj, dict):
        return obj2str(list(obj.items()))
    elif hasattr(obj, "identifier"):
        return obj2str(obj.identifier)
    else:
        raise ValueError("No rule to convert object '{}' to string.".
                         format(obj.__class__))
