"""Compress .rtdc files"""
from __future__ import annotations

import argparse
import atexit
import multiprocessing as mp
import pathlib
import threading
import warnings

import hdf5plugin
import h5py

from ..definitions.feat_const import FEATURES_IMAGE_ROI
from ..rtdc_dataset import RTDCWriter, new_dataset
from ..rtdc_dataset.copier import get_size, rtdc_copy
from ..rtdc_dataset.feat_basin.chop_images import write_chopped_images
from .. import util
from .._version import version

from . import common


def compress(
        path_in: str | pathlib.Path | None = None,
        path_out: str | pathlib.Path | None = None,
        crop_event_images: bool = False,
        check_suffix: bool = True,
        ret_path: bool = False,
        ):
    """Create a new dataset with all features compressed lossless

    Parameters
    ----------
    path_in: str or pathlib.Path
        file to compress
    path_out: str or pathlib.Path
        output file path
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

    # command log
    logs = {"dclab-compress": common.get_command_log(paths=[path_in])}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with h5py.File(path_in, locking=False) as h5, \
                h5py.File(path_temp, "w") as hc:
            bytes_total = mp.Value("Q")
            bytes_written = mp.Value("Q")
            stop_event = threading.Event()

            monitor_thread = threading.Thread(
                target=common.monitor,
                args=("Compression", bytes_total, bytes_written, stop_event),
                name="Compression",
                daemon=True)
            monitor_thread.start()

            features_copy = (
                list(h5.get("events", {}).keys())
                + list(h5.get("basin_events", {}).keys()))

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
                    if feat in h5.get("events", {}):
                        # only crop image features that are not a basin already
                        features_copy.remove(feat)
                        features_crop.append(feat)
                        bytes_total.value += get_size(h5[f"events/{feat}"])

            rtdc_copy(src_h5file=h5,
                      dst_h5file=hc,
                      features=features_copy,
                      include_basins=True,
                      include_logs=True,
                      include_tables=True,
                      meta_prefix="",
                      bytes_total=bytes_total,
                      bytes_written=bytes_written,
                      )

            for feat in features_crop:
                with new_dataset(path_in) as ds:
                    write_chopped_images(
                        ds=ds,
                        feat=feat,
                        h5_dst=hc,
                        bytes_chopped=bytes_written,
                        )

            stop_event.set()
            monitor_thread.join()

            hc.require_group("logs")
            # rename old dclab-compress logs
            for lkey in ["dclab-compress", "dclab-compress-warnings"]:
                if lkey in hc["logs"]:
                    # This is cached, so no worry calling it multiple times.
                    md55m = util.hashfile(path_in, count=80)
                    # rename
                    hc["logs"][f"{lkey}_{md55m}"] = hc["logs"][lkey]
                    del hc["logs"][lkey]

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

    if ret_path:
        return path_out
    else:
        return None


def compress_parser():
    descr = "Create a compressed version of an .rtdc file. This can be " \
            + "used for saving disk space (loss-less compression). The " \
            + "data generated during an experiment is usually not compressed."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('input', metavar="INPUT", type=str,
                        help='Input path (.rtdc file)')
    parser.add_argument('output', metavar="OUTPUT", type=str,
                        help='Output path (.rtdc file)')
    parser.add_argument("--crop-event-images",
                        dest="crop_event_images",
                        action="store_true",
                        help="Reduce the amount of storage required by "
                             "cropping event images. This removes pixels "
                             "not related to events which reduced file size.")
    parser.set_defaults(crop_event_images=False)
    parser.add_argument('--version', action='version',
                        version=f'dclab-compress {version}')
    return parser
