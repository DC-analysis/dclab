#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Load RT-DC datasets"""
from __future__ import unicode_literals

import pathlib
import warnings

from ..compat import str_types

from .core import RTDCBase
from . import fmt_dict, fmt_dcor, fmt_hdf5, fmt_tdms, fmt_hierarchy


def check_dataset(path_or_ds):
    """deprecated, to not use"""
    warnings.warn("Please use dclab.rtdc_dataset.check.check_dataset!",
                  DeprecationWarning)
    from . import check  # avoid circular import
    return check.check_dataset(path_or_ds)


def load_file(path, identifier=None, **kwargs):
    path = pathlib.Path(path).resolve()
    for fmt in [fmt_hdf5.RTDC_HDF5, fmt_tdms.RTDC_TDMS]:
        if fmt.can_open(path):
            return fmt(path, identifier=identifier, **kwargs)
    else:
        raise ValueError("Unknown file format: '{}'".format(path.suffix))


def new_dataset(data, identifier=None, **kwargs):
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
        - DCOR resource URL
    identifier: str
        A unique identifier for this dataset. If set to `None`
        an identifier is generated.
    kwargs: dict
        Additional parameters passed to the RTDCBase subclass

    Returns
    -------
    dataset: subclass of :class:`dclab.rtdc_dataset.RTDCBase`
        A new dataset instance
    """
    if isinstance(data, dict):
        return fmt_dict.RTDC_Dict(data, identifier=identifier, **kwargs)
    elif isinstance(data, (str_types)) or isinstance(data, pathlib.Path):
        if (isinstance(data, pathlib.Path)
                or (not data.count("://")  # prevent Pathing it on Windows
                    and pathlib.Path(data).exists())):
            return load_file(data, identifier=identifier, **kwargs)
        else:
            return fmt_dcor.RTDC_DCOR(data, identifier=identifier, **kwargs)
    elif isinstance(data, RTDCBase):
        return fmt_hierarchy.RTDC_Hierarchy(data, identifier=identifier,
                                            **kwargs)
    else:
        msg = "data type not supported: {}".format(data.__class__)
        raise NotImplementedError(msg)
