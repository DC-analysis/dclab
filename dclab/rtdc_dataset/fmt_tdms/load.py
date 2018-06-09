#!/usr/bin/python
# -*- coding: utf-8 -*-
from nptdms import TdmsFile


def wrap_tdmsfile(path):
    """A unicode-safe wrapper for opening tdms files

    This can be removed once moved to Python 3.
    """
    try:  # ideal case
        data = TdmsFile(str(path))
    except UnicodeDecodeError:  # probably Python 2
        try:
            data = TdmsFile(unicode(path))
        except BaseException:  # also Python 2
            try:
                data = TdmsFile(unicode(path).encode("utf-8"))
            except BaseException:
                data = TdmsFile(str(path).decode("utf-8"))
    return data
