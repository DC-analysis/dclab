#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Check RT-DC data sets for completeness"""
from __future__ import unicode_literals

import pathlib
import sys

import h5py

from .core import RTDCBase
from . import fmt_dict, fmt_hdf5, fmt_tdms, fmt_hierarchy

from .. import definitions as dfn


if sys.version_info[0] == 2:
    str_classes = (str, unicode)
else:
    str_classes = str


#: keys that must be present for every measurement
IMPORTANT_KEYS = {
    "experiment": [
        "date",
        "event count",
        "run index",
        "sample",
        "time"],
    "imaging": [
        "frame rate",
        "pixel size",
        "roi position x",
        "roi position y",
        "roi size x",
        "roi size y"],
    "setup": [
        "channel width",
        "chip region",
        "flow rate",
        "medium"],
}

#: keys that must be present for fluorescence measurements
IMPORTANT_KEYS_FL = {
    "fluorescence": [
        "bit depth",
        "channel count",
        "laser 1 power",
        "laser 2 power",
        "laser 3 power",
        "laser 1 lambda",
        "laser 2 lambda",
        "laser 3 lambda",
        "sample rate",
        "signal max",
        "signal min",
        "trace median"],
}

#: maximum line length in log files
LOG_MAX_LINE_LENGTH = 100


class HDF5ImageMetaDataMissing(BaseException):
    """Used for making sure that hdf5 images can be opened in hdfview"""
    pass


class HDF5LogLineLengthTooLong(BaseException):
    """Used for making sure that hdf5 log file lines are not too long"""
    pass


class RTDCFeatureMissingError(BaseException):
    """Used for missing features"""
    pass


class RTDCMetaDataMissingError(BaseException):
    """Used for missing meta data"""
    pass


class RTDCMetaDataTypeError(BaseException):
    """Used for missing meta data"""
    pass


class RTDCUnknownMetaKeyError(BaseException):
    """Used for unknown meta data keys"""
    pass


def check_dataset(path_or_ds):
    """Check if a dataset is complete

    Parameters
    ----------
    path_or_ds: str or RTDCBase
        Full path to a data set on disk or an instance of RTDCBase

    Raises
    ------
    various errors
    """
    if not isinstance(path_or_ds, RTDCBase):
        ds = load_file(path_or_ds)
    else:
        ds = path_or_ds
    # check for meta data types
    bad_meta = []
    for sec in ds.config:
        for key in ds.config[sec]:
            if sec in dfn.CFG_ANALYSIS:
                pass
            elif (sec not in dfn.config_types or
                  key not in dfn.config_types[sec]):
                msg = "[{}] {} in {}".format(sec, key, path_or_ds)
                raise RTDCUnknownMetaKeyError(msg)
            elif not isinstance(ds.config[sec][key],
                                dfn.config_types[sec][key]):
                msg = "[{}] {} = {}, ({} vs. {})".format(
                    sec,
                    key,
                    ds.config[sec][key],
                    dfn.config_types[sec][key],
                    type(ds.config[sec][key]))
                bad_meta.append(msg)
    if bad_meta:
        msg = "Wrong meta data types in {}: \n".format(path_or_ds) \
              + "\n".join(bad_meta)
        raise RTDCMetaDataTypeError(msg)
    # check important meta data keys
    miss_meta = []
    tocheck = IMPORTANT_KEYS
    # should we also check for fluorescence keys?
    if ("fluorescence" in ds.config or
        "fl1_max" in ds._events or
        "fl2_max" in ds._events or
            "fl3_max" in ds._events):
        tocheck = tocheck.copy()
        tocheck.update(IMPORTANT_KEYS_FL)
    # search for missing keys
    for sec in tocheck:
        if sec not in ds.config:
            msg = "No section: {}".format(sec)
            miss_meta.append(msg)
        else:
            for key in tocheck[sec]:
                if key not in ds.config[sec]:
                    msg = "No key [{}] {}".format(sec, key)
                    miss_meta.append(msg)
    if miss_meta:
        msg = "Missing meta data in {}: \n".format(path_or_ds) \
              + "\n".join(miss_meta)
        raise RTDCMetaDataMissingError(msg)
    # check for feature column names
    ukwn_feats = []
    for feat in ds._events.keys():
        if feat not in dfn.feature_names + ["contour", "image", "trace"]:
            ukwn_feats.append(feat)
    if ukwn_feats:
        msg = "Unknown feature names in {}: \n".format(path_or_ds) \
              + "\n".join(ukwn_feats)
    # hdf5-based checks
    if ds.format == "hdf5":
        # check meta data of images
        if "image" in ds._events:
            imdat = ds["image"]
            miss_imkey = []
            for key, val in [['CLASS', 'IMAGE'],
                             ['IMAGE_VERSION', '1.2'],
                             ['IMAGE_SUBCLASS', 'IMAGE_GRAYSCALE']]:
                if key not in imdat.attrs or imdat.attrs.get(key) != val:
                    miss_imkey.append(key)
            if miss_imkey:
                msg = "Missing image attrs: {}".format(",".join(miss_imkey))
                raise HDF5ImageMetaDataMissing(msg)
        # check length of logs
        bad_lines = []
        with h5py.File(ds.path, mode="r") as h5:
            logs = h5["logs"]
            for logname in logs.keys():
                log = logs[logname]
                for ii in range(len(log)):
                    if len(log[ii]) > LOG_MAX_LINE_LENGTH:
                        bad_lines.append("{}: line {}".format(logname, ii))
        if bad_lines:
            msg = "Line length exceeds {} ".format(LOG_MAX_LINE_LENGTH) \
                  + " in {}: \n".format(path_or_ds) \
                  + "\n".join(bad_lines)
            raise HDF5LogLineLengthTooLong(msg)


def load_file(path, identifier=None):
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
        return load_file(data, identifier=identifier)
    elif isinstance(data, RTDCBase):
        return fmt_hierarchy.RTDC_Hierarchy(data, identifier=identifier)
    else:
        msg = "data type not supported: {}".format(data.__class__)
        raise NotImplementedError(msg)
