#!/usr/bin/python
# -*- coding: utf-8 -*-
"""RT-DC file format writer"""
from __future__ import unicode_literals

import copy

import h5py
import numpy as np

from ..compat import hdf5_str
from .. import definitions as dfn
from .._version import version


def store_contour(h5group, data, compression):
    if not isinstance(data, (list, tuple)):
        # single event
        data = [data]
    grp = h5group.require_group("contour")
    curid = len(grp.keys())
    for ii, cc in enumerate(data):
        grp.create_dataset("{}".format(curid + ii),
                           data=cc,
                           fletcher32=True,
                           compression=compression)


def store_image(h5group, data, compression):
    if len(data.shape) == 2:
        # single event
        data = data.reshape(1, data.shape[0], data.shape[1])
    if "image" not in h5group:
        maxshape = (None, data.shape[1], data.shape[2])
        chunks = (100, data.shape[1], data.shape[2])
        dset = h5group.create_dataset("image",
                                      data=data,
                                      dtype=np.uint8,
                                      maxshape=maxshape,
                                      chunks=chunks,
                                      fletcher32=True,
                                      compression=compression)
        # Create and Set image attributes:
        # HDFView recognizes this as a series of images.
        # Use np.string_ as per
        # http://docs.h5py.org/en/stable/strings.html#compatibility
        dset.attrs.create('CLASS', np.string_('IMAGE'))
        dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
        dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))
    else:
        dset = h5group["image"]
        oldsize = dset.shape[0]
        dset.resize(oldsize + data.shape[0], axis=0)
        dset[oldsize:] = data


def store_mask(h5group, data, compression):
    # store binary mask data as uint8 to allow visualization in HDFView
    data = np.asarray(data, dtype=np.uint8)
    if data.max() != 255 and data.max() != 0 and data.min() == 0:
        data = data / data.max() * 255
    if len(data.shape) == 2:
        # single event
        data = data.reshape(1, data.shape[0], data.shape[1])
    if "mask" not in h5group:
        maxshape = (None, data.shape[1], data.shape[2])
        chunks = (100, data.shape[1], data.shape[2])
        dset = h5group.create_dataset("mask",
                                      data=data,
                                      dtype=np.uint8,
                                      maxshape=maxshape,
                                      chunks=chunks,
                                      fletcher32=True,
                                      compression=compression)
        # Create and Set image attributes
        # HDFView recognizes this as a series of images
        dset.attrs.create('CLASS', np.string_('IMAGE'))
        dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
        dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))
    else:
        dset = h5group["mask"]
        oldsize = dset.shape[0]
        dset.resize(oldsize + data.shape[0], axis=0)
        dset[oldsize:] = data


def store_scalar(h5group, name, data, compression):
    if np.isscalar(data):
        # single event
        data = np.atleast_1d(data)
    if name not in h5group:
        h5group.create_dataset(name,
                               data=data,
                               maxshape=(None,),
                               chunks=(100,),
                               fletcher32=True,
                               compression=compression
                               )
    else:
        dset = h5group[name]
        oldsize = dset.shape[0]
        dset.resize(oldsize + data.shape[0], axis=0)
        dset[oldsize:] = data


def store_trace(h5group, data, compression):
    firstkey = sorted(list(data.keys()))[0]
    if len(data[firstkey].shape) == 1:
        # single event
        for dd in data:
            data[dd] = data[dd].reshape(1, -1)
    # create trace group
    grp = h5group.require_group("trace")

    for flt in data:
        # create traces datasets
        if flt not in grp:
            maxshape = (None, data[flt].shape[1])
            chunks = (100, data[flt].shape[1])
            grp.create_dataset(flt,
                               data=data[flt],
                               maxshape=maxshape,
                               chunks=chunks,
                               fletcher32=True,
                               compression=compression)
        else:
            dset = grp[flt]
            oldsize = dset.shape[0]
            dset.resize(oldsize + data[flt].shape[0], axis=0)
            dset[oldsize:] = data[flt]


def write(path_or_h5file, data={}, meta={}, logs={}, mode="reset",
          compression=None):
    """Write data to an RT-DC file

    Parameters
    ----------
    path: path or h5py.File
        The path or the hdf5 file object to write to.
    data: dict-like
        The data to store. Each key of `data` must be a valid
        feature name (see :func:`dclab.dfn.feature_exists`). The
        data type must be given according to the feature type:

        - scalar feature: 1d ndarray of size `N`, any dtype,
          with the number of events `N`.
        - contour: list of `N` 2d ndarrays of shape `(2,C)`, any dtype,
          with each ndarray containing the x- and y- coordinates
          of `C` contour points in pixels.
        - image: 3d ndarray of shape `(N,A,B)`, is converted to uint8,
          with the image dimensions `(x,y) = (A,B)`
        - mask: 3d ndarray of shape `(N,A,B)`, is converted to bool,
          with the image dimensions `(x,y) = (A,B)`
        - trace: 2d ndarray of shape `(N,T)`, any dtype
          with a globally constant trace length `T`.
    meta: dict of dicts
        The meta data to store (see `dclab.dfn.config_keys`).
        Each key depicts a meta data section name whose data is given
        as a dictionary, e.g.

            meta = {"imaging": {"exposure time": 20,
                                "flash duration": 2,
                                ...
                                },
                    "setup": {"channel width": 20,
                              "chip region": "channel",
                              ...
                              },
                    ...
                    }

        Only section key names and key values therein registered
        in dclab are allowed and are converted to the pre-defined
        dtype.
    logs: dict of lists
        Each key of `logs` refers to a list of strings that contains
        logging information. Each item in the list can be considered to
        be one line in the log file.
    mode: str
        Defines how the input `data` and `logs` are stored:
        - "append": append new data to existing Datasets; the opened
                    `h5py.File` object is returned (used in real-
                    time data storage)
        - "replace": replace keys given by `data` and `logs`; the
                    opened `h5py.File` object is closed and `None`
                    is returned (used for ancillary feature storage)
        - "reset": do not keep any previous data; the opened
                   `h5py.File` object is closed and `None` is returned
                   (default)
    compression: str
        Compression method for "contour", "image", and "trace" data
        as well as logs; one of [None, "lzf", "gzip", "szip"].

    Notes
    -----
    If `data` is an instance of RTDCBase, then `meta` must be set to
    `data.config`, otherwise no meta data will be saved.
    """
    if mode not in ["append", "replace", "reset"]:
        raise ValueError("`mode` must be one of [append, replace, reset]")
    if compression not in [None, "gzip", "lzf", "szip"]:
        raise ValueError("`compression` must be one of [gzip, lzf, szip]")

    # make sure we are not overriding anything
    meta = copy.deepcopy(meta)

    if (not hasattr(data, "__iter__") or
        not hasattr(data, "__contains__") or
            not hasattr(data, "__getitem__") or
            isinstance(data, (list, np.ndarray))):
        msg = "`data` must be dict-like"
        raise ValueError(msg)

    # Check meta data
    for sec in meta:
        if sec not in dfn.config_keys:
            # only allow writing of meta data that are not editable
            # by the user (not dclab.dfn.CFG_ANALYSIS)
            msg = "Meta data section not defined in dclab: {}".format(sec)
            raise ValueError(msg)
        for ck in meta[sec]:
            if ck not in dfn.config_keys[sec]:
                msg = "Meta key not defined in dclab: {}:{}".format(sec, ck)
                raise ValueError(msg)

    # Check feature keys
    feat_keys = []
    for kk in data:
        if dfn.feature_exists(kk):
            feat_keys.append(kk)
        else:
            raise ValueError("Unknown key '{}'!".format(kk))
        # verify trace names
        if kk == "trace":
            for sk in data["trace"]:
                if sk not in dfn.FLUOR_TRACES:
                    msg = "Unknown trace key: {}".format(sk)
                    raise ValueError(msg)

    # Create file
    # (this should happen after all checks)
    if isinstance(path_or_h5file, h5py.File):
        h5obj = path_or_h5file
    else:
        if mode == "reset":
            h5mode = "w"
        else:
            h5mode = "a"
        h5obj = h5py.File(path_or_h5file, mode=h5mode)

    # update version
    # - if it is not already in the hdf5 file (prevent override)
    # - if it is explicitly given in meta (append to old version string)
    if ("setup:software version" not in h5obj.attrs
            or ("setup" in meta and "software version" in meta["setup"])):
        thisver = "dclab {}".format(version)
        if "setup" in meta and "software version" in meta["setup"]:
            oldver = meta["setup"]["software version"]
            thisver = "{} | {}".format(oldver, thisver)
        if "setup" not in meta:
            meta["setup"] = {}
        meta["setup"]["software version"] = thisver
    # Write meta
    for sec in meta:
        for ck in meta[sec]:
            idk = "{}:{}".format(sec, ck)
            conffunc = dfn.config_funcs[sec][ck]
            if conffunc is str:
                h5obj.attrs[idk] = np.string_(meta[sec][ck].encode("utf-8"))
            else:
                h5obj.attrs[idk] = conffunc(meta[sec][ck])

    # Write data
    # create events group
    events = h5obj.require_group("events")
    # remove previous data
    if mode == "replace":
        for rk in feat_keys:
            if rk in events:
                del events[rk]
    # store experimental data
    for fk in feat_keys:
        if dfn.scalar_feature_exists(fk):
            store_scalar(h5group=events,
                         name=fk,
                         data=data[fk],
                         compression=compression)
        elif fk == "contour":
            store_contour(h5group=events,
                          data=data["contour"],
                          compression=compression)
        elif fk == "image":
            store_image(h5group=events,
                        data=data["image"],
                        compression=compression)
        elif fk == "mask":
            store_mask(h5group=events,
                       data=data["mask"],
                       compression=compression)
        elif fk == "trace":
            store_trace(h5group=events,
                        data=data["trace"],
                        compression=compression)

    # Write logs
    if logs:
        log_group = h5obj.require_group("logs")
        # remove previous data
        if mode == "replace":
            for rl in logs:
                if rl in log_group:
                    del log_group[rl]
        dt = h5py.special_dtype(vlen=hdf5_str)
        for lkey in logs:
            ldata = logs[lkey]
            if isinstance(ldata, (str, hdf5_str)):
                # single event
                ldata = [ldata]
            lnum = len(ldata)
            if lkey not in log_group:
                log_dset = log_group.create_dataset(lkey,
                                                    (lnum,),
                                                    dtype=dt,
                                                    maxshape=(None,),
                                                    chunks=True,
                                                    fletcher32=True,
                                                    compression=compression)
                for ii, line in enumerate(ldata):
                    log_dset[ii] = line
            else:
                log_dset = log_group[lkey]
                oldsize = log_dset.shape[0]
                log_dset.resize(oldsize + lnum, axis=0)
                for ii, line in enumerate(ldata):
                    log_dset[oldsize + ii] = line

    if mode == "append":
        return h5obj
    else:
        h5obj.close()
        return None
