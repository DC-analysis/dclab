#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals 

import sys
import warnings

from .core import hashfile, RTDCBase
from .config import Configuration
from . import fmt_dict, fmt_tdms, fmt_hierarchy

if sys.version_info[0] == 2:
    str_classes = (str, unicode)
else:
    str_classes = str


def new_dataset(data):
    """Initialize a new RT-DC data set
    
    Parameters
    ----------
    data:
        can be one of the following:
        - dict
        - .tdms file
        - subclass of `RTDCBase`
    
    Returns
    -------
    dataset: subclass of `RTDCBase`
        A new dataset instance
    """
    if isinstance(data, dict):
        return fmt_dict.RTDC_Dict(data)
    elif isinstance(data, str_classes):
        return fmt_tdms.RTDC_TDMS(data)
    elif isinstance(data, RTDCBase):
        return fmt_hierarchy.RTDC_Hierarchy(data)
    else:
        msg = "data type not supported: {}".format(data.__class__)
        raise NotImplementedError(msg)


def RTDC_DataSet(*args, **kwargs):
    msg = "Please use `new_dataset` instead of `RTDC_DataSet` (DEPRECATION)!"
    warnings.warn(msg)
    if args:
        return new_dataset(args[0])
    else:
        keys = list(kwargs.keys())
        return new_dataset(kwargs[keys[0]])
