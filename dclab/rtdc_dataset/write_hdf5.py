"""RT-DC file format writer"""
import copy

import h5py
import numpy as np

from .. import definitions as dfn
from .._version import version

from .writer import RTDCWriter


#: Chunk size for storing HDF5 data
CHUNK_SIZE = 100


def write(path_or_h5file, data=None, meta=None, logs=None, mode="reset",
          compression=None):
    """Write data to an RT-DC file

    Parameters
    ----------
    path_or_h5file: path or h5py.File
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
    if logs is None:
        logs = {}
    if meta is None:
        meta = {}
    if data is None:
        data = {}
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
        if sec == "user":
            # user-defined metadata are always written.
            # Any errors (incompatibilities with HDF5 attributes)
            # are the user's responsibility
            continue
        elif sec not in dfn.CFG_METADATA:
            # only allow writing of meta data that are not editable
            # by the user (not dclab.dfn.CFG_ANALYSIS)
            msg = "Meta data section not defined in dclab: {}".format(sec)
            raise ValueError(msg)
        for ck in meta[sec]:
            if not dfn.config_key_exists(sec, ck):
                raise ValueError(f"Meta key not defined in dclab: {sec}:{ck}")

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
            value = meta[sec][ck]
            if isinstance(value, bytes):
                # We never store byte attribute values.
                # In this case, `conffunc` should be `str` or `lcstr` or
                # somesuch. But we don't test that, because no other datatype
                # competes with str for bytes.
                value = value.decode("utf-8")
            if sec == "user":
                # store user-defined metadata as-is
                h5obj.attrs[idk] = value
            else:
                # pipe the metadata through the hard-coded converter functions
                convfunc = dfn.get_config_value_func(sec, ck)
                h5obj.attrs[idk] = convfunc(value)

    # Write data
    hw = RTDCWriter(h5obj, mode=mode, compression=compression)

    # store experimental data
    for fk in feat_keys:
        hw.store_feature(fk, data[fk])

    # Write logs
    if logs:
        for lkey in logs:
            hw.store_log(lkey, logs[lkey])

    if mode == "append":
        return h5obj
    else:
        h5obj.close()
        return None
