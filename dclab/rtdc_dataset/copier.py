"""Helper methods for copying .rtdc data"""
from __future__ import annotations

import json
import multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized
import re
from typing import Literal

import h5py
import h5py.h5o
import hdf5plugin
import numpy as np

from ..definitions import feature_exists, scalar_feature_exists
from ..external.packaging import parse as parse_version
from ..util import hashobj

from .fmt_hdf5 import DEFECTIVE_FEATURES, RTDC_HDF5
from .writer import RTDCWriter


class ByteBookKeeper:
    def __init__(self):
        """A book keeper for sharing copying state between threads

        This is only process-safe, if all pages are defined before
        spawning the new subprocesses.
        """
        self._book = {}

    def __setitem__(self, page, size):
        if page in self._book:
            self._book[page][0].value = size
        else:
            self._book[page] = [mp.Value("Q", size), mp.Value("Q", 0)]

    def __getitem__(self, page):
        return self._book[page]

    def remove(self, page):
        """Remove a page from the book (e.g. invalid feature)"""
        self._book.pop(page)

    def update(self, page, total=0, completed=0):
        """Update an empty book keeping page"""
        self[page] = total
        self[page][1].value = completed

    def complete_item(self, page):
        """Set a page as completed"""
        self._book[page][1].value = self._book[page][0].value

    def get_total(self):
        """Return sum of total byte count for all pages"""
        return sum(page[0].value for page in self._book.values())

    def get_done(self):
        """Return sum of partial byte count for all pages"""
        return sum(page[1].value for page in self._book.values())

    def get_progress(self) -> float:
        """Return total progress in interval [0, 1]"""
        return self.get_done() / (self.get_total() or 1e6)


def rtdc_copy(src_h5file: h5py.Group,
              dst_h5file: h5py.Group,
              features: list[str] | Literal['all', 'scalar', 'none'] = "all",
              include_basins: bool = True,
              include_logs: bool = True,
              include_tables: bool = True,
              meta_prefix: str = "",
              byte_book_keeper: ByteBookKeeper | None = None,
              ):
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
    byte_book_keeper:
        If specified, will be used to track the progress of copying features.
        The basin *definitions* are not included due to their variable size.
        Logs are also not included, because the line length may vary.
    """
    bbk = byte_book_keeper

    # identify features in source file
    if "events" in src_h5file:
        events_src = (list(src_h5file.get("events", {}).keys())
                      + list(src_h5file.get("basin_events", {}).keys()))
    else:
        events_src = []

    # actual features to copy
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

    feature_iter = [feat for feat in set(feature_iter) if feature_exists(feat)]

    # determine the total size of the data that need to be copied
    if bbk is not None:
        if include_tables and "tables" in src_h5file:
            bbk["tables"] = get_size(src_h5file["tables"])

        for feat in feature_iter:
            for loc in ["events", "basin_events"]:
                if feat in src_h5file[loc]:
                    bbk[feat] = get_size(src_h5file[loc][feat])
                    break

    # copy internal basin data (this will also update the byte_book_keeper)
    basin_feat = internal_basin_events_copy(
        src_h5file=src_h5file,
        dst_h5file=dst_h5file,
        features=feature_iter,
        byte_book_keeper=bbk,
    )

    # remove features that are already written to the output file
    for feat in basin_feat:
        if feat in feature_iter:
            feature_iter.remove(feat)

    bn_regexp = re.compile("^basinmap[0-9]*$")  # future-proof regexp
    src_basin_feats = [f for f in events_src if bn_regexp.match(f)]
    if include_basins:
        # Explicitly include all remaining basinmap features.
        for feat in src_basin_feats:
            if feat not in feature_iter and feat not in basin_feat:
                feature_iter.append(feat)
    else:
        # Remove all remaining basinmap features from the list.
        for feat in src_basin_feats:
            if feat in feature_iter:
                feature_iter.remove(feat)

    # copy metadata (bytes not counted)
    dst_h5file.attrs.update(src_h5file.attrs)

    # copy logs (bytes not counted)
    if include_logs and "logs" in src_h5file:
        dst_h5file.require_group("logs")
        for l_key in src_h5file["logs"]:
            h5ds_copy(src_loc=src_h5file["logs"],
                      src_name=l_key,
                      dst_loc=dst_h5file["logs"],
                      dst_name=meta_prefix + l_key,
                      recursive=False)

    # copy basin definitions (bytes not counted)
    if include_basins and "basins" in src_h5file:
        basin_definition_copy(src_h5file=src_h5file,
                              dst_h5file=dst_h5file,
                              features_iter=feature_iter)

    # copy tables
    if include_tables and "tables" in src_h5file:
        hdf5_version = h5py.version.hdf5_version
        if parse_version(hdf5_version) < parse_version("1.14.4"):
            # There was a problem with h5copy in some rare situations.
            # https://github.com/HDFGroup/hdf5/issues/3214
            raise ImportError("Your version of h5py is built against HDF5 "
                              f"{hdf5_version}, but dclab requires HDF5 "
                              f"1.14.4 for writing tables.")

        dst_h5file.require_group("tables")
        for tkey in src_h5file["tables"]:
            h5ds_copy(
                src_loc=src_h5file["tables"],
                src_name=tkey,
                dst_loc=dst_h5file["tables"],
                dst_name=meta_prefix + tkey,
                recursive=False,
                bytes_copied=bbk["tables"][1] if bbk is not None else None,
                )

    # copy regular event features
    if feature_iter:
        src_events = src_h5file["events"]
        dst_events = dst_h5file.require_group("events")
        for feat in feature_iter:
            if feat in src_events:
                # Skip all defective features. These are features that
                # are known to be invalid (e.g. ancillary features that
                # were computed falsely) and must be recomputed by dclab.
                if feat in DEFECTIVE_FEATURES:
                    defective = DEFECTIVE_FEATURES[feat](src_h5file)
                    if defective:
                        if bbk is not None:
                            bbk.remove(feat)
                        # Do not copy this feature
                        continue

                if bbk is not None:
                    bbk[feat] = get_size(src_events[feat])

                dst = h5ds_copy(
                    src_loc=src_events,
                    src_name=feat,
                    dst_loc=dst_events,
                    recursive=True,
                    bytes_copied=bbk[feat][1] if bbk is not None else None,
                    )

                if scalar_feature_exists(feat):
                    # complement min/max values for all scalar features
                    for ufunc, attr in [(np.nanmin, "min"),
                                        (np.nanmax, "max"),
                                        (np.nanmean, "mean"),
                                        ]:
                        if attr not in dst.attrs:
                            dst.attrs[attr] = ufunc(dst)


def internal_basin_events_copy(
        src_h5file: h5py.Group,
        dst_h5file: h5py.Group,
        features: list[str],
        byte_book_keeper: ByteBookKeeper | None = None,
        ) -> list[str]:
    """Copy internal basin data from the input to the output file

    The basin dictionaries are read and only the `basinmap` features
    that are required are copied to the output file.
    """
    bbk = byte_book_keeper
    basin_feat = []

    bn_dicts = RTDC_HDF5.basin_get_dicts_from_h5file(src_h5file)

    for bn in bn_dicts:
        if bn["type"] == "internal":
            bn_feats = []
            for feat in bn["features"]:
                if feat in features and f"basin_events/{feat}" in src_h5file:
                    if bbk is not None:
                        bbk.update(
                            page=feat,
                            total=get_size(src_h5file[f"basin_events/{feat}"]),
                            completed=0,
                        )
                        bytes_copied = bbk[feat][1]
                    else:
                        bytes_copied = None
                    bn_feats.append(feat)
                    h5ds_copy(
                        src_loc=src_h5file["basin_events"],
                        src_name=feat,
                        dst_loc=dst_h5file.require_group("basin_events"),
                        dst_name=feat,
                        bytes_copied=bytes_copied,
                        )

            # Complete by copying basinmap features and basin definitions
            if bn_feats:
                # Note down features that we added
                basin_feat += bn_feats
                # Write basinmap feature
                if bn["mapping"].startswith("basinmap"):
                    feat = bn["mapping"]
                    basin_feat.append(feat)
                    if bbk is not None:
                        bbk[feat] = get_size(src_h5file[f"events/{feat}"])
                    h5ds_copy(
                        src_loc=src_h5file["events"],
                        src_name=feat,
                        dst_loc=dst_h5file.require_group("events"),
                        bytes_copied=bbk[feat][1] if bbk else None,
                    )
                # Rewrite basin definition
                bn["features"] = bn_feats
                # Convert edited `bn` to JSON and write feature data
                b_lines = json.dumps(bn, indent=2).split("\n")
                key = hashobj(b_lines)
                if key not in dst_h5file.require_group("basins"):
                    with RTDCWriter(dst_h5file) as hw:
                        hw.write_text(dst_h5file["basins"], key, b_lines)
    return list(set(basin_feat))


def basin_definition_copy(src_h5file, dst_h5file, features_iter):
    """Copy basin definitions `src_h5file["basins"]` to the new file

    Normally, we would just use :func:`h5ds_copy` to copy basins from
    one dataset to another. However, if we are e.g. only copying scalar
    features, and there are non-scalar features in the internal basin,
    then we must rewrite the basin definition of the internal basin.

    The `features_iter` list of features defines which features are
    relevant for the internal basin.

    To copy internal basins, use :func:`internal_basin_events_copy`.
    """
    dst_h5file.require_group("basins")
    # Load the basin information
    basin_dicts = RTDC_HDF5.basin_get_dicts_from_h5file(src_h5file)
    for bn in basin_dicts:
        if bn["type"] == "internal":
            # handled in `internal_basin_events_copy`
            continue

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

        h5ds_copy(src_loc=src_h5file["basins"],
                  src_name=b_key,
                  dst_loc=dst_h5file["basins"],
                  dst_name=b_key,
                  recursive=False)


def h5ds_copy(src_loc: h5py.Group,
              src_name: str,
              dst_loc: h5py.Group,
              dst_name: str | None = None,
              ensure_compression: bool = True,
              recursive: bool = True,
              bytes_copied: Synchronized[int] | None = None,
              ):
    """Copy an HDF5 Dataset from one group to another

    Parameters
    ----------
    src_loc: h5py.Group
        The source location
    src_name: str
        Name of the dataset in `src_loc`
    dst_loc: h5py.Group
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
    bytes_copied: mp.Value
        A shared :class:`multiprocessing.Value` instance to which
        the number of bytes written is added during the copying process;
        Use this if you would like to track the progress.

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
            elif src.chunks is None:
                # The input dataset does not have chunks defined
                chunks = RTDCWriter.get_best_nd_chunks(
                    item_shape=src.shape[1:],
                    item_dtype=src.dtype
                    )
                if chunks[0] > len(src):
                    chunks = (len(src),) + chunks[1:]
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

            # Make sure chunks are always well-defined.
            assert chunks is not None

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
                # Write to the new dataset in chunks that we defined during
                # dataset creation.
                chunk_size = chunks[0]
                num_chunks = int(np.ceil(len(src) / chunk_size))
                for ii in range(num_chunks):
                    ch_slice = slice(chunk_size*ii, chunk_size*(ii+1))
                    new_chunk = src[ch_slice]
                    dst[ch_slice] = new_chunk
                    if bytes_copied is not None:
                        bytes_copied.value += new_chunk.nbytes

            # Also write all the attributes
            dst.attrs.update(src.attrs)
        else:
            # Copy the Dataset to the destination as-is.
            h5py.h5o.copy(src_loc=src_loc.id,
                          src_name=src_name.encode(),
                          dst_loc=dst_loc.id,
                          dst_name=dst_name.encode(),
                          )
            if bytes_copied is not None:
                bytes_copied.value += src.nbytes

    elif recursive and isinstance(src, h5py.Group):
        dst_rec = dst_loc.require_group(dst_name)
        dst_rec.attrs.update(src.attrs)
        for key in src:
            h5ds_copy(src_loc=src,
                      src_name=key,
                      dst_loc=dst_rec,
                      ensure_compression=ensure_compression,
                      recursive=recursive,
                      bytes_copied=bytes_copied,
                      )
    else:
        raise ValueError(f"The object {src_name} in {src.file} is not "
                         f"a dataset!")
    return dst_loc[dst_name]


def get_size(h5_obj: h5py.Group | h5py.Dataset | list | tuple | None
             ) -> int:
    """Recursively return the size of an HDF5 object (group or dataset)

    Returns
    -------
    size
        the size in bytes
    """
    size = 0
    if isinstance(h5_obj, h5py.Group):
        for item in h5_obj.values():
            size += get_size(item)
    elif isinstance(h5_obj, h5py.Dataset):
        size += h5_obj.nbytes
    elif isinstance(h5_obj, (tuple, list)):
        for item in h5_obj:
            size += get_size(item)
    elif h5_obj is None:
        pass
    return size


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
