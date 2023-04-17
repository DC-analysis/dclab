"""Helper methods for copying .rtdc data"""
from __future__ import annotations

from typing import Literal

import h5py
import h5py.h5o
import hdf5plugin
import numpy as np

from ..definitions import feature_exists, scalar_feature_exists


def rtdc_copy(src_h5file: h5py.Group,
              dst_h5file: h5py.Group,
              features: Literal['all', 'scalar', 'none'] = "all",
              include_logs: bool = True,
              include_tables: bool = True,
              meta_prefix: str = ""):
    """Create a compressed copy of an RT-DC file"""
    # metadata
    for akey in src_h5file.attrs:
        dst_h5file.attrs[akey] = src_h5file.attrs[akey]

    # logs
    if include_logs and "logs" in src_h5file:
        dst_h5file.require_group("logs")
        for lkey in src_h5file["logs"]:
            h5ds_copy(src_loc=src_h5file["logs"],
                      src_name=lkey,
                      dst_loc=dst_h5file["logs"],
                      dst_name=meta_prefix + lkey)

    # tables
    if include_tables and "tables" in src_h5file:
        dst_h5file.require_group("tables")
        for tkey in src_h5file["tables"]:
            h5ds_copy(src_loc=src_h5file["tables"],
                      src_name=tkey,
                      dst_loc=dst_h5file["tables"],
                      dst_name=meta_prefix + tkey)
    # features
    if features != "none":
        scalar_only = features == "scalar"
        dst_h5file.require_group("events")
        for feat in src_h5file["events"]:
            if feature_exists(feat, scalar_only=scalar_only):
                dst = h5ds_copy(src_loc=src_h5file["events"],
                                src_name=feat,
                                dst_loc=dst_h5file["events"])
                if scalar_feature_exists(feat):
                    # complement min/max values for all scalar features
                    for ufunc, attr in [(np.nanmin, "min"),
                                        (np.nanmax, "max"),
                                        (np.nanmean, "mean"),
                                        ]:
                        if attr not in dst.attrs:
                            dst.attrs[attr] = ufunc(dst)


def h5ds_copy(src_loc, src_name, dst_loc, dst_name=None,
              ensure_compression=True):
    """Copy an HDF5 Dataset from one group to another

    Parameters
    ----------
    src_loc: h5py.H5Group
        The source location
    src_name: str
        Name of the dataset in `src_loc`
    dst_loc: h5py.H5Group
        The destination location
    dst_name: str
        The name of the destination dataset, defaults to `src_name`
    ensure_compression: bool
        Whether to make sure that the data are compressed,
        If disabled, then all data from the source will be
        just copied and not compressed.

    Returns
    -------
    dst: h5py.Dataset
        The dataset `dst_loc[dst_name]`

    Raises
    ------
    ValueError:
        If the named source is not a h5py.Dataset
    """
    compression_kwargs = hdf5plugin.Zstd(clevel=5)
    dst_name = dst_name or src_name
    src = src_loc[src_name]
    if isinstance(src, h5py.Dataset):
        if ensure_compression and not is_properly_compressed(src):
            # Manually create a compressed version of the dataset.
            dst = dst_loc.create_dataset(name=dst_name,
                                         shape=src.shape,
                                         dtype=src.dtype,
                                         chunks=src.chunks,
                                         fletcher32=True,
                                         **compression_kwargs
                                         )
            for chunk in src.iter_chunks():
                dst[chunk] = src[chunk]
            # Also write all the attributes
            for key in src.attrs:
                dst.attrs[key] = src.attrs[key]
        else:
            # Copy the Dataset to the destination as-is.
            h5py.h5o.copy(src_loc=src_loc.id,
                          src_name=src_name,
                          dst_loc=dst_loc,
                          dst_name=dst_name,
                          )
    else:
        raise ValueError(f"The object {src_name} in {src.file} is not "
                         f"a dataset!")
    return dst_loc[dst_name]


def is_properly_compressed(h5obj):
    """Check whether an HDF5 object is properly compressed

    The compression check only returns True if the input file was
    compressed with the Zstandard compression using compression
    level 5 or higher.
    """
    # Since version 0.43.0, we use Zstandard compression
    # which does not show up in the `compression`
    # attribute of `obj`.
    create_plist = h5obj.id.get_create_plist()
    filter_args = create_plist.get_filter_by_id(32015)
    if filter_args is not None and filter_args[1][0] >= 5:
        properly_compressed = True
    else:
        properly_compressed = False
    return properly_compressed
