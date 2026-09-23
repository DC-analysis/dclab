"""Compress .rtdc files"""
from __future__ import annotations

import argparse
import atexit
import pathlib
import threading
import warnings

import hdf5plugin
import h5py
import numpy as np

from ..definitions.feat_const import FEATURES_IMAGE_ROI
from ..rtdc_dataset import RTDCWriter, new_dataset
from ..rtdc_dataset.copier import ByteBookKeeper, get_size, rtdc_copy
from ..rtdc_dataset.feat_basin.chop_images import write_chopped_images
from ..rtdc_dataset.feat_basin.basin_proxy import BasinProxy

from .. import util
from .._version import version

from . import common


def compress(
        path_in: str | pathlib.Path | None = None,
        path_out: str | pathlib.Path | None = None,
        basin_input: list[pathlib.Path] | list[str] | None = None,
        crop_event_images: bool = False,
        check_suffix: bool = True,
        ret_path: bool = False,
        byte_book_keeper: ByteBookKeeper | None = None,
        ):
    """Create a new dataset with all features compressed lossless

    Parameters
    ----------
    path_in: str or pathlib.Path
        file to compress
    path_out: str or pathlib.Path
        output file path
    basin_input:
        list of additional paths from which to put features in the
        output file
    crop_event_images: bool
        Crop event images using an internal "h5datasetchop" basin.
        This reduces storage size by removing pixels that are not
        related to an event. For this to work, the features "mask",
        "frame", "size_y", and "size_x", as well as "image_bg" for the
        "image" feature must be available.
    check_suffix: bool
        check suffixes for input and output files
    ret_path: bool
        whether to return the output path
    byte_book_keeper:
        for progress monitoring, kwarg used for testing only

    Returns
    -------
    path_out: pathlib.Path (optional)
        output path (with possibly corrected suffix)
    """
    cmp_kw = hdf5plugin.Zstd(clevel=5)
    if path_out is None or path_in is None:
        parser = compress_parser()
        args = parser.parse_args()
        path_in = args.input
        path_out = args.output
        basin_input = args.basin_input
        crop_event_images = args.crop_event_images

    # setup paths
    # input
    assert path_in is not None
    path_in = pathlib.Path(path_in)
    if check_suffix and path_in.suffix != ".rtdc":
        raise ValueError(f"Unsupported file type: '{path_in.suffix}'")
    # output
    assert path_out is not None
    path_out = pathlib.Path(path_out)
    if path_out.suffix != ".rtdc":
        path_out = path_out.with_name(path_out.name + ".rtdc")
    path_out.unlink(missing_ok=True)
    # temporary
    path_temp = path_out.with_suffix(".rtdc~")
    path_temp.unlink(missing_ok=True)
    atexit.register(path_temp.unlink, missing_ok=True)
    # basin paths (resolve for later comparison)
    basin_input = [pathlib.Path(pp).resolve() for pp in basin_input or []]

    # command log
    logs = {"dclab-compress": common.get_command_log(paths=[path_in])}

    # book keeper alias
    bbk = byte_book_keeper
    if bbk is None:
        bbk = ByteBookKeeper()
    bbk["cli-compress"] = 1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Prepare monitoring
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=common.monitor,
            args=("Compression", bbk, stop_event),
            name="Compression",
            daemon=True)
        monitor_thread.start()

        # Make sure all basin files are valid and available
        # and create `basin_maps` dictionary with mapping for each basin.
        basin_maps = {}
        if basin_input:
            with new_dataset(path_in) as ds0:
                for bn in ds0.basins:
                    if (bn.basin_type == "file"
                        and bn.basin_format == "hdf5"
                            and bn.is_available()):
                        pb = pathlib.Path(bn.location).resolve()
                        if pb in basin_input:
                            if bn.mapping == "same":
                                basin_maps[pb] = None
                            else:
                                basin_maps[pb] = ds0[bn.mapping][:]

        if set(basin_maps.keys()) != set(basin_input):
            raise ValueError(
                f"The following files are not basins of the input file "
                f"'{path_in.name}': {set(basin_input) - set(basin_maps)}"
            )

        # Which features are taken from which file?
        features_all = set()
        pp_features = {}
        paths = [path_in] + list(basin_maps.keys())
        for pp in paths:
            pp_feats = set()
            with h5py.File(pp, locking=False) as h5:
                for loc in ["events", "basin_events"]:
                    for feat in h5.get(loc, {}):
                        if feat not in features_all:
                            pp_feats.add(feat)
                            if bbk is not None:
                                bbk[feat] = get_size(h5[f"{loc}/{feat}"])
            # Set of features in `features` but not `features_all`
            pp_features[pp] = list(pp_feats)
            # update features_all list
            features_all = features_all | pp_feats

        # Start with the actual compression
        with h5py.File(path_temp, "w") as h5_out:
            for pp in paths:
                with h5py.File(pp, locking=False) as h5_in:
                    compress_input(
                        h5_in=h5_in,
                        h5_out=h5_out,
                        features=pp_features[pp],
                        crop_event_images=crop_event_images,
                        basinmap=basin_maps.get(pp),
                        byte_book_keeper=bbk,
                        include_other_data=(pp == path_in),
                    )

            h5_out.require_group("logs")
            # rename old dclab-compress logs
            for lkey in ["dclab-compress", "dclab-compress-warnings"]:
                if lkey in h5_out["logs"]:
                    # This is cached, so no worry calling it multiple times.
                    md55m = util.hashfile(path_in, count=80)
                    # rename
                    h5_out["logs"][f"{lkey}_{md55m}"] = h5_out["logs"][lkey]
                    del h5_out["logs"][lkey]

        # warnings log
        if w:
            logs["dclab-compress-warnings"] = common.assemble_warnings(w)

    # Write log file
    with RTDCWriter(path_temp,
                    compression_kwargs=cmp_kw,
                    mode="append") as hw:
        for name, value in logs.items():
            hw.store_log(name, value)

    # Finally, rename temp to out
    path_temp.rename(path_out)

    if bbk is not None:
        bbk.complete_item("cli-compress")
    stop_event.set()
    monitor_thread.join()

    if ret_path:
        return path_out
    else:
        return None


def compress_input(h5_in: h5py.File,
                   h5_out: h5py.File,
                   features: list[str],
                   crop_event_images: bool = False,
                   basinmap: np.ndarray | None = None,
                   include_other_data: bool = False,
                   byte_book_keeper: ByteBookKeeper | None = None,
                   ):
    """Copy/compress data from one HDF5 file to another

    Parameters
    ----------
    h5_in
        Input file
    h5_out
        Output file
    features
        List of features to copy from one file to the other
    crop_event_images
        Whether to chop images via `rtdc_dataset.feat_basin.chop_images`
    basinmap
        If None, just copy all data from the input to the output. If
        set to an integer array, this is the basin mapping that should
        be used for exporting the data. If `basinmap` is defined, no
        other data (logs, basin definitions, tables) are written.
    include_other_data
        Whether to include logs, tables, and basins in the output file.
        This only makes sense for the original input file.
    byte_book_keeper
        Optional for tracking the progress
    """
    bbk = byte_book_keeper

    features_crop = []
    if crop_event_images:
        feat_croppable = [f[0] for f in FEATURES_IMAGE_ROI]
        # 'image_bg' must stay intact (for embedding of "image")
        # and in addition, it should actually be stored as a
        # regular basin feature (mapping multiple images to one)
        feat_croppable.remove("image_bg")
        # 'mask' should not be chopped, because the "background"
        # of mask data is just zeros. Mask data are best compressed
        # with regular compression algorithms (e.g. zstd). For a
        # full blood measurement, chopping up the mask reduces the
        # file size by less than only 10 MB. Compression time of
        # the mask feature would increase by a factor of ~6x.
        feat_croppable.remove("mask")

        for feat in feat_croppable:
            if feat in h5_in.get("events", {}):
                # only crop image features that are not a basin already
                features.remove(feat)
                features_crop.append(feat)

    if basinmap is None:
        # Just copy HDF5 data
        rtdc_copy(src_h5file=h5_in,
                  dst_h5file=h5_out,
                  features=features,
                  include_basins=include_other_data,
                  include_logs=include_other_data,
                  include_tables=include_other_data,
                  meta_prefix="",
                  byte_book_keeper=bbk,
                  )
    else:
        # Instantiate an RTDCBase object. This is more time-consuming.
        # Note that we don't write logs, tables, or basins when `basinmap` is
        # defined, because in this case we are only interested in the
        # feature data.
        assert not include_other_data, "basin should not write logs, etc."
        assert basinmap.dtype.kind in "iu", "must be integer array"
        with new_dataset(h5_in.filename) as ds, RTDCWriter(h5_out) as hw:
            ds_used = BasinProxy(ds, basinmap=basinmap)
            for feat in features:
                hw.store_feature(feat, ds_used[feat])
                if bbk is not None:
                    bbk.complete_item(feat)

    for feat in features_crop:
        with new_dataset(h5_in.filename) as ds:
            if basinmap is None:
                # Take all data
                ds_used = ds
            else:
                # Take filtered events
                ds_used = BasinProxy(ds, basinmap=basinmap)
            if bbk is not None:
                bbk.update(
                    page=feat,
                    total=(np.prod(ds_used[feat][0].shape)
                           * len(ds_used)
                           * ds_used[feat].dtype.itemsize),
                    completed=0,
                    )

            # Note that image cropping requires the features
            # ["frame", "mask", "size_x", "size_y", "image_bg"].
            # Usually, these are taken from either the input `ds`
            # or the output file `h5_out`. If you get an error
            # about missing features, make sure that the features
            # exist and that the basin file that has them comes
            # before the basin with the image data.
            write_chopped_images(
                ds=ds_used,
                feat=feat,
                h5_dst=h5_out,
                bytes_chopped=bbk[feat][1] if bbk is not None else None,
                )
            if bbk is not None:
                bbk.complete_item(feat)


def compress_parser():
    descr = "Create a compressed version of an .rtdc file. This can be " \
            + "used for saving disk space (loss-less compression). The " \
            + "data generated during an experiment is usually not compressed."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument("input", metavar="INPUT", type=str,
                        help="Input path (.rtdc file)")
    parser.add_argument("output", metavar="OUTPUT", type=str,
                        help="Output path (.rtdc file)")
    parser.add_argument("--crop-event-images",
                        dest="crop_event_images",
                        action="store_true",
                        help="Reduce the amount of storage required by "
                             "cropping event images. This removes pixels "
                             "not related to events which reduced file size.")
    parser.add_argument("-b",
                        "--basin-input",
                        action="append",
                        type=str,
                        help="additional basin input file from which to "
                             "include features in the output file.")
    parser.set_defaults(crop_event_images=False)
    parser.add_argument("--version", action="version",
                        version=f"dclab-compress {version}")
    return parser
