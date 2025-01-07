"""Helper methods for copying .rtdc data"""
from __future__ import annotations

import json
import re
from typing import List, Literal

import h5py
import h5py.h5o
import hdf5plugin
import numpy as np

from ..definitions import feature_exists, scalar_feature_exists
from ..util import hashobj

from .fmt_hdf5 import DEFECTIVE_FEATURES, RTDC_HDF5
from .writer import RTDCWriter


def rtdc_copy(src_h5file: h5py.Group,
              dst_h5file: h5py.Group,
              features: List[str] | Literal['all', 'scalar', 'none'] = "all",
              include_basins: bool = True,
              include_logs: bool = True,
              include_tables: bool = True,
              meta_prefix: str = ""):
    """Create a compressed copy of an RT-DC file

    Parameters
    ----------
    src_h5file: h5py.Group
        Input HDF5 file
    dst_h5file: h5py.Group
        Output HDF5 file
    features: list of strings or one of ['all', 'scalar', 'none']
        If this is a list then it specifies the features that are copied from
        `src_h5file` to `dst_h5file`. Alternatively, you may specify 'all'
        (copy all features), 'scalar' (copy only scalar features), or 'none'
        (don't copy any features).
    include_basins: bool
        Copy the basin information from `src_h5file` to `dst_h5file`.
    include_logs: bool
        Copy the logs from `src_h5file` to `dst_h5file`.
    include_tables: bool
        Copy the tables from `src_h5file` to `dst_h5file`.
    meta_prefix: str
        Add this prefix to the name of the logs and tables in `dst_h5file`.
    """
    # metadata
    dst_h5file.attrs.update(src_h5file.attrs)

    # events in source file
    if "events" in src_h5file:
        events_src = list(src_h5file["events"].keys())
    else:
        events_src = []

    if include_basins and "basin_events" in src_h5file:
        events_src += list(src_h5file["basin_events"].keys())
        events_src = sorted(set(events_src))

    # logs
    if include_logs and "logs" in src_h5file:
        dst_h5file.require_group("logs")
        for l_key in src_h5file["logs"]:
            h5ds_copy(src_loc=src_h5file["logs"],
                      src_name=l_key,
                      dst_loc=dst_h5file["logs"],
                      dst_name=meta_prefix + l_key,
                      recursive=False)

    # tables
    if include_tables and "tables" in src_h5file:
        dst_h5file.require_group("tables")
        for tkey in src_h5file["tables"]:
            # There appears to be a problem with h5copy in some rare
            # situations, so we do not use h5copy, but read and write
            # the table data directly.
            # https://github.com/HDFGroup/hdf5/issues/3214
            # The following caused a Segmentation fault:
            # h5ds_copy(src_loc=src_h5file["tables"],
            #           src_name=tkey,
            #           dst_loc=dst_h5file["tables"],
            #           dst_name=meta_prefix + tkey,
            #           recursive=False)
            copy_table = dst_h5file["tables"].create_dataset(
                name=tkey,
                data=src_h5file["tables"][tkey][:],
                fletcher32=True,
                **hdf5plugin.Zstd(clevel=5))
            copy_table.attrs.update(src_h5file["tables"][tkey].attrs)

    # events
    if isinstance(features, list):
        feature_iter = features
    elif features == "all":
        feature_iter = events_src
    elif features == "scalar":
        feature_iter = [feat for feat in events_src
                        if feature_exists(feat, scalar_only=True)]
    elif features == "none":
        feature_iter = []
    else:
        raise ValueError(f"`features` must be either a list of feature names "
                         f"or one of 'all', 'scalar' or 'none', got "
                         f"'{features}'")

    # Additional check for basin features.
    bn_regexp = re.compile("^basinmap[0-9]*$")  # future-proof regexp
    src_basin_feats = [f for f in events_src if bn_regexp.match(f)]
    if include_basins:
        # Make sure all 'basinmap?' features are included in the output file.
        for feat in src_basin_feats:
            if feat not in feature_iter:
                feature_iter.append(feat)
    else:
        # We do not need the basinmap features, because basins are
        # stripped from the output file.
        for feat in src_basin_feats:
            if feat in feature_iter:
                feature_iter.remove(feat)

    # copy basin definitions
    if include_basins and "basins" in src_h5file:
        basin_definition_copy(src_h5file=src_h5file,
                              dst_h5file=dst_h5file,
                              features_iter=feature_iter)

    if feature_iter:
        dst_h5file.require_group("events")
        for feat in feature_iter:
            if not feature_exists(feat):
                continue
            elif feat in src_h5file["events"]:
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

            elif (include_basins
                    and "basin_events" in src_h5file
                    and feat in src_h5file["basin_events"]):
                # Also copy internal basins which should have been defined
                # in the "basin_events" group.
                if feat in src_h5file["basin_events"]:
                    h5ds_copy(src_loc=src_h5file["basin_events"],
                              src_name=feat,
                              dst_loc=dst_h5file.require_group("basin_events"),
                              dst_name=feat
                              )


def basin_definition_copy(src_h5file, dst_h5file, features_iter):
    """Copy basin definitions `src_h5file["basins"]` to the new file

    Normally, we would just use :func:`h5ds_copy` to copy basins from
    one dataset to another. However, if we are e.g. only copying scalar
    features, and there are non-scalar features in the internal basin,
    then we must rewrite the basin definition of the internal basin.

    The `features_iter` list of features defines which features are
    relevant for the internal basin.
    """
    dst_h5file.require_group("basins")
    # Load the basin information
    basin_dicts = RTDC_HDF5.basin_get_dicts_from_h5file(src_h5file)
    for bn in basin_dicts:
        b_key = bn["key"]

        if b_key in dst_h5file["basins"]:
            # already stored therein
            continue

        # sanity check
        if b_key not in src_h5file["basins"]:
            raise ValueError(
                f"Failed to parse basin information correctly. Source file "
                f"{src_h5file} does not contain basin {b_key} which I got "
                f"from `RTDC_HDF5.basin_get_dicts_from_h5file`.")

        if bn["type"] == "internal":
            # Make sure we define the internal features selected
            feat_used = [f for f in bn["features"] if f in features_iter]
            if len(feat_used) == 0:
                # We don't have any internal features, don't write anything
                continue
            elif feat_used != bn["features"]:
                bn["features"] = feat_used
                rewrite = True
            else:
                rewrite = False
        else:
            # We do not have an internal basin, just copy everything
            rewrite = False

        if rewrite:
            # Convert edited `bn` to JSON and write feature data
            b_lines = json.dumps(bn, indent=2).split("\n")
            key = hashobj(b_lines)
            if key not in dst_h5file["basins"]:
                with RTDCWriter(dst_h5file) as hw:
                    hw.write_text(dst_h5file["basins"], key, b_lines)
        else:
            # copy only
            h5ds_copy(src_loc=src_h5file["basins"],
                      src_name=b_key,
                      dst_loc=dst_h5file["basins"],
                      dst_name=b_key,
                      recursive=False)


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
            dst.attrs.update(src.attrs)
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
