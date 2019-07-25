#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Load RT-DC datasets for completeness"""
from __future__ import unicode_literals

import pathlib
import warnings

from ..compat import str_types

from .core import RTDCBase
from . import fmt_dict, fmt_hdf5, fmt_tdms, fmt_hierarchy


def check_dataset(path_or_ds):
    """deprecated, to not use"""
    warnings.warn("Please use dclab.rtdc_dataset.check.check_dataset!",
                  DeprecationWarning)
    from . import check  # avoid circular import
    return check.check_dataset(path_or_ds)


def load_file(path, identifier=None):
    path = pathlib.Path(path).resolve()
    if path.suffix == ".tdms":
        return fmt_tdms.RTDC_TDMS(path, identifier=identifier)
    elif path.suffix == ".rtdc":
        return fmt_hdf5.RTDC_HDF5(path, identifier=identifier)
    else:
        raise ValueError("Unknown file extension: '{}'".format(path.suffix))


def new_dataset(data, identifier=None):
    """Initialize a new RT-DC dataset

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
        A unique identifier for this dataset. If set to `None`
        an identifier is generated.

    Returns
    -------
    dataset: subclass of :class:`dclab.rtdc_dataset.RTDCBase`
        A new dataset instance
    """
    if isinstance(data, dict):
        return fmt_dict.RTDC_Dict(data, identifier=identifier)
    elif isinstance(data, (str_types)) or isinstance(data, pathlib.Path):
        return load_file(data, identifier=identifier)
    elif isinstance(data, RTDCBase):
        return fmt_hierarchy.RTDC_Hierarchy(data, identifier=identifier)
    else:
        msg = "data type not supported: {}".format(data.__class__)
        raise NotImplementedError(msg)
