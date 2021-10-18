"""Legacy RT-DC file format writer"""
import warnings

import numpy as np

from .writer import RTDCWriter, CHUNK_SIZE  # noqa: F401


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
    warnings.warn("`write` is deptecated; please use the dclab.RTDCWriter "
                  "class for writing .rtdc data to disk.",
                  DeprecationWarning)
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

    if (not hasattr(data, "__iter__") or
        not hasattr(data, "__contains__") or
            not hasattr(data, "__getitem__") or
            isinstance(data, (list, np.ndarray))):
        msg = "`data` must be dict-like"
        raise ValueError(msg)

    # Initialize writer
    hw = RTDCWriter(path_or_h5file, mode=mode, compression=compression)
    # Metadata
    hw.store_metadata(meta)
    # Features
    for fk in data:
        hw.store_feature(fk, data[fk])
    # Logs
    if logs:
        for lkey in logs:
            hw.store_log(lkey, logs[lkey])

    # Return HDF5 object or close it
    if mode == "append":
        return hw.h5file
    else:
        hw.h5file.close()
        return None
