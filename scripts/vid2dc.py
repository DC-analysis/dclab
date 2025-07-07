"""vid2dc

Convert video files to .rtdc files. This script generates a valid .rtdc
file from a video file (e.g. an .avi file). This script does not perform
segmentation and feature extraction, a feature that is available via the
ChipStream package (https://github.com/DC-analysis/ChipStream).

Prerequisites:

    pip install dclab[tdms] imageio[pyav] click

Author: Paul Müller
License: GPLv3+
"""
import configparser
import logging
import pathlib
import time
import traceback

import click
from dclab.rtdc_dataset import fmt_tdms
from dclab.util import hashfile, hashobj
from dclab import RTDCWriter
import h5py
import imageio.v3 as iio
import numpy as np


logger = logging.getLogger("vid2dc")
logging.basicConfig(level=logging.INFO)


__version__ = "2025.07.07"


@click.command()
@click.argument("path_in",
                type=click.Path(exists=True,
                                dir_okay=True,
                                resolve_path=True,
                                path_type=pathlib.Path))
@click.argument("path-out",
                required=False,
                type=click.Path(dir_okay=False,
                                writable=True,
                                resolve_path=True,
                                path_type=pathlib.Path),
                )
def vid2dc(path_in: pathlib.Path,
           path_out: pathlib.Path | None = None):
    """Convert raw video data to raw .rtdc data

    Use this command to convert a video file (e.g. from a legacy
    .tdms measurement or from a custom setup) to a raw .rtdc file.
    No segmentation or feature extraction will take place.

    You may also specify an input directory instead of a single file
    for batch conversion.
    """
    if path_in.is_dir():
        if path_out is not None:
            raise NotImplementedError("If you pass an input directory, you "
                                      "must omit the output directory.")
        for ext in [".avi", ".mp4", ".mpg", ".wmv", ".mkv"]:
            for pp in path_in.rglob(f"*{ext}"):
                try:
                    convert_video_to_dc(video_path=pp)
                except BaseException:
                    elog = pp.with_name(pp.stem + "_vid2dc_error.log")
                    elog.write_text(traceback.format_exc())
                    click.secho(f"Could not process {pp}, wrote error log "
                                f"to {elog}.",
                                bold=True, fg="red")
    else:
        convert_video_to_dc(video_path=path_in, dc_path=path_out)


def convert_video_to_dc(video_path: str | pathlib.Path,
                        dc_path: str | pathlib.Path = None,
                        ):
    """Convert an .avi file to an .rtdc file

    A valid .rtdc file is produced. Metadata are guessed, but you
    can put metadata in camera.ini files. This is not fully supported yet.
    Please check the metadata of the output .rtdc file.

    Parameters
    ----------
    video_path:
        Path to a video file (e.g. an .avi file) to convert to .rtdc.
    dc_path:
        Path to an .rtdc file that is created from the video file.
        If the .rtdc file already exists, it will only be overridden
        if the input file or metadata changed.
    """
    video_path = pathlib.Path(video_path)
    if dc_path is None:
        dc_path = video_path.with_name(video_path.stem + "_raw.rtdc")
    dc_path = pathlib.Path(dc_path)

    # get the dataset configuration
    config, log_paths = fmt_tdms.RTDC_TDMS.extract_tdms_config(
        video_path, ret_source_files=True, ignore_missing=True)

    # RecoTeem stores an ini file next to the measurement
    prt = video_path.with_suffix(".ini")
    if prt.exists():
        log_paths.append(prt)

    # read logs
    logs = {}
    for ppl in log_paths:
        logs[ppl.name] = ppl.read_text().split("\n")

    vmeta = iio.improps(video_path)
    width = vmeta.shape[2]
    height = vmeta.shape[1]

    config["imaging"].setdefault("frame rate", 2000)
    config["imaging"].setdefault("pixel size", 0.34)
    config["imaging"]["roi size x"] = width
    config["imaging"]["roi size y"] = height

    # extract additional information from an optional ini file (RecoTeem)
    if prt.exists():
        # measurement index
        if prt.name.startswith("M") and prt.name.count("_"):
            mid = video_path.name.split("_")[0]
            config["experiment"]["run index"] = int(mid[1:])
        cp = configparser.RawConfigParser(allow_no_value=True)
        # Allow upper-case keys (comment for sample name)
        cp.optionxform = lambda x: x
        cp.read(prt)
        rt_fr = cp.getint("set parameter", "frame rate [fps]", fallback=None)
        # frame rate
        if rt_fr is not None:
            config["imaging"]["frame rate"] = rt_fr
        rt_dur = cp.getfloat("set parameter", "set record time [s]",
                             fallback=None)
        # duration
        if rt_dur is not None:
            config["online_filter"]["target duration"] = rt_dur / 60
        # sample name
        if cp.has_section("Optional Text"):
            descr = list(cp["Optional Text"].keys())[0]
            config["experiment"]["sample"] = descr

    # Set the time of the input file
    tse = video_path.stat().st_mtime
    loct = time.localtime(tse)
    # Start time of measurement ('HH:MM:SS')
    timestr = time.strftime("%H:%M:%S", loct)
    config["experiment"].setdefault("time", timestr)
    # Date of measurement ('YYYY-MM-DD')
    datestr = time.strftime("%Y-%m-%d", loct)
    config["experiment"].setdefault("date", datestr)
    config["experiment"].setdefault("sample", video_path.name)

    config["setup"].setdefault("chip region", "channel")
    config["setup"].setdefault("flow rate", 0)
    config["setup"].setdefault("software version", f"vid2dc {__version__}")

    config = dict(config)
    if "filtering" in config:
        config.pop("filtering")

    # Compute a unique hash of the input files
    video_hash = hashobj([hashobj(config),
                          hashfile(video_path),
                          video_path.stat().st_size,
                          ])

    # If the output file already exists, check whether the hashes match.
    log_name = "vid2dc"
    if dc_path.exists():
        with h5py.File(dc_path) as h5:
            if (log_name in h5.get("logs", [])
                    and h5["logs"][log_name].attrs.get("hash") == video_hash):
                # We can reuse this file
                logger.info(f"Reusing existing file {dc_path}")
                return
            else:
                logger.info(f"Rewriting existing {dc_path} (hash mismatch)")
                dc_path.unlink()

    vid = iio.imiter(video_path)

    # This is the actual workload of this function. Populate the .rtdc
    # file with the image data.
    logger.info(f"Writing .rtdc file {dc_path}")
    with RTDCWriter(dc_path, mode="reset") as hw:
        # store the image data to the output file in chunks
        chunk_size = hw.get_best_nd_chunks(item_shape=(height, width),
                                           item_dtype=np.uint8)[0]
        image_chunk = np.zeros((chunk_size, height, width), dtype=np.uint8)
        ii = 0
        for image in vid:
            if len(image.shape) == 3:
                image = image[:, :, 0]
            image_chunk[ii] = image
            ii += 1
            if ii == chunk_size:
                ii = 0
                hw.store_feature("image", image_chunk)
        # store the rest
        if ii:
            hw.store_feature("image", image_chunk[:ii])

        event_count = hw.h5file["events/image"].shape[0]
        config["experiment"]["event count"] = event_count
        hw.store_metadata(config)

        # store the input file information as a log
        hw.store_log(
            name=log_name,
            lines=[
                f"Input filename: {video_path.name}",
                f"Output filename: {dc_path.name}",
                f"Input file size: {video_path.stat().st_size}",
                f"Input last edited: {video_path.stat().st_mtime}",
                f"Unique conversion hash: {video_hash}",
            ]
        )

        # store time
        hw.store_feature(
            "time",
            np.arange(event_count) / config["imaging"]["frame rate"])
        # index starts at 1
        hw.store_feature("index", np.arange(1, event_count + 1))
        # frame is an ascending number, just start with 1
        hw.store_feature("frame", np.arange(1, event_count + 1))

        # only store hash attribute when successful
        hw.h5file["logs"][log_name].attrs["hash"] = video_hash


if __name__ == "__main__":
    vid2dc()
