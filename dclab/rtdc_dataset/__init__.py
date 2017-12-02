#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals 

import pathlib
import sys
import warnings

from .config import Configuration
from .core import RTDCBase
from . import fmt_dict, fmt_hdf5, fmt_tdms, fmt_hierarchy
from .util import hashfile
from .write_hdf5 import write


if sys.version_info[0] == 2:
    str_classes = (str, unicode)
else:
    str_classes = str


def _load_file(path, identifier):
    path = pathlib.Path(path).resolve()
    if path.suffix == ".tdms":
        return fmt_tdms.RTDC_TDMS(str(path), identifier=identifier)
    elif path.suffix == ".rtdc":
        return fmt_hdf5.RTDC_HDF5(str(path), identifier=identifier)
    else:
        raise ValueError("Unknown file extension: '{}'".format(path.suffix))


def new_dataset(data, identifier=None):
    """Initialize a new RT-DC data set
    
    Parameters
    ----------
    data:
        can be one of the following:
        - dict
        - .tdms file
        - .rtdc file
        - subclass of `RTDCBase`
          (will create a hierarchy child)
    identifier: str
        A unique identifier for this data set. If set to `None`
        an identifier will be generated.
    
    Returns
    -------
    dataset: subclass of `RTDCBase`
        A new dataset instance
    """
    if isinstance(data, dict):
        return fmt_dict.RTDC_Dict(data, identifier=identifier)
    elif isinstance(data, (str_classes)) or isinstance(data, pathlib.Path):
        return _load_file(data, identifier=identifier)
    elif isinstance(data, RTDCBase):
        return fmt_hierarchy.RTDC_Hierarchy(data, identifier=identifier)
    else:
        msg = "data type not supported: {}".format(data.__class__)
        raise NotImplementedError(msg)
