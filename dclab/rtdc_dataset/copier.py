"""Helper methods for copying .rtdc data"""
from __future__ import annotations

from typing import Literal

import h5py
import h5py.h5o
import hdf5plugin
import numpy as np

from ..definitions import feature_exists, scalar_feature_exists

from .fmt_hdf5 import DEFECTIVE_FEATURES


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
                      dst_name=meta_prefix + lkey,
                      recursive=False)

    # tables
    if include_tables and "tables" in src_h5file:
        dst_h5file.require_group("tables")
        for tkey in src_h5file["tables"]:
            h5ds_copy(src_loc=src_h5file["tables"],
                      src_name=tkey,
                      dst_loc=dst_h5file["tables"],
                      dst_name=meta_prefix + tkey,
                      recursive=False)

    # events
    if features != "none":
        scalar_only = features == "scalar"
        dst_h5file.require_group("events")
        for feat in src_h5file["events"]:
            if feature_exists(feat, scalar_only=scalar_only):
                # Skip all defective features. These are features that
                # are known to be invalid (e.g. ancillary features that
                # were computed falsely) and must be recomputed by dclab.
                if feat in DEFECTIVE_FEATURES:
                    defective = DEFECTIVE_FEATURES[feat](src_h5file)
                    if defective:
                        continue

                dst = h5ds_copy(src_loc=src_h5file["events"],
                                src_name=feat,
                                dst_loc=dst_h5file["events"],
                                recursive=True)
                if scalar_feature_exists(feat):
                    # complement min/max values for all scalar features
                    for ufunc, attr in [(np.nanmin, "min"),
                                        (np.nanmax, "max"),
                                        (np.nanmean, "mean"),
                                        ]:
                        if attr not in dst.attrs:
                            dst.attrs[attr] = ufunc(dst)


def h5ds_copy(src_loc, src_name, dst_loc, dst_name=None,
              ensure_compression=True, recursive=True):
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
    recursive: bool
        Whether to recurse into HDF5 Groups (this is required e.g.
        for copying the "trace" feature)

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
            # Chunk size larger than dataset size is not allowed
            # in h5py's `make_new_dset`.
            if src.shape[0] == 0:
                # Ignore empty datasets (This sometimes happens with logs).
                return
            elif src.chunks and src.chunks[0] > src.shape[0]:
                # The chunks in the input file are larger than the dataset
                # shape. So we set the chunks to the shape. Here, we only
                # check for the first axis (event count for feature data),
                # because if the chunks vary in any other dimension then
                # there is something fundamentally wrong with the input
                # dataset (which we don't want to endorse, and where there
                # could potentially be a lot of data put into ram).
                chunks = list(src.chunks)
                chunks[0] = src.shape[0]
                chunks = tuple(chunks)
            else:
                # original chunk size is fine
                chunks = src.chunks
            # Variable length strings, compression, and fletcher32 are not
            # a good combination. If we encounter any logs, then we have
            # to write them with fixed-length strings.
            # https://forum.hdfgroup.org/t/fletcher32-filter-on-variable-
            # length-string-datasets-not-suitable-for-filters/9038/4
            if src.dtype.kind == "O":
                # We are looking at logs with variable length strings.
                max_length = max([len(ii) for ii in src] + [100])
                dtype = f"S{max_length}"
                convert_to_s_fixed = True
            else:
                dtype = src.dtype
                convert_to_s_fixed = False

            # Manually create a compressed version of the dataset.
            dst = dst_loc.create_dataset(name=dst_name,
                                         shape=src.shape,
                                         dtype=dtype,
                                         chunks=chunks,
                                         fletcher32=True,
                                         **compression_kwargs
                                         )
            if convert_to_s_fixed:
                # We are looking at old variable-length log strings.
                dst[:] = src[:].astype(dtype)
            elif chunks is None:
                dst[:] = src[:]
            else:
                for chunk in src.iter_chunks():
                    dst[chunk] = src[chunk]
            # Also write all the attributes
            for key in src.attrs:
                dst.attrs[key] = src.attrs[key]
        else:
            # Copy the Dataset to the destination as-is.
            h5py.h5o.copy(src_loc=src_loc.id,
                          src_name=src_name.encode(),
                          dst_loc=dst_loc.id,
                          dst_name=dst_name.encode(),
                          )
    elif recursive and isinstance(src, h5py.Group):
        dst_rec = dst_loc.require_group(dst_name)
        for key in src:
            h5ds_copy(src_loc=src,
                      src_name=key,
                      dst_loc=dst_rec,
                      ensure_compression=ensure_compression,
                      recursive=recursive)
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
