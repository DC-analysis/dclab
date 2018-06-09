#!/usr/bin/python
# -*- coding: utf-8 -*-
import pathlib
import tempfile

from nptdms import TdmsFile

def wrap_tdmsfile(path):
    """A unicode-safe wrapper for loading tdms files

    TdmsFile accepts a string object (not unicode-safe) or an open
    file (problems under Windows with Anaconda).

    This workaround creates a temporary symlink and loads the data
    from there.
    """
    path = pathlib.Path(path)
    tpath = tempfile.mktemp(prefix="dclab_nptdms_workaround_", suffix=".tdms")
    tpath = pathlib.Path(tpath)
    tpath.symlink_to(path)
    data = TdmsFile(str(tpath))
    tpath.unlink()
    return data
