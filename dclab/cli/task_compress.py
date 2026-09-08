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

from ..rtdc_dataset import rtdc_copy, RTDCWriter
from .. import util
from .._version import version

from . import common


def compress(
        path_in: str | pathlib.Path | None = None,
        path_out: str | pathlib.Path | None = None,
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

            rtdc_copy(src_h5file=h5,
                      dst_h5file=hc,
                      features="all",
                      include_basins=True,
                      include_logs=True,
                      include_tables=True,
                      meta_prefix="",
                      bytes_total=bytes_total,
                      bytes_written=bytes_written,
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
    parser.set_defaults(force=False)
    parser.add_argument('--version', action='version',
                        version=f'dclab-compress {version}')
    return parser
