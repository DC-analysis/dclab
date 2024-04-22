"""Repack (similar to h5repack) .rtdc files"""
from __future__ import annotations

import argparse
import pathlib

import h5py

from ..rtdc_dataset import rtdc_copy
from .._version import version

from . import common


def repack(
        path_in: str | pathlib.Path = None,
        path_out: str | pathlib.Path = None,
        strip_basins: bool = False,
        strip_logs: bool = False,
        check_suffix: bool = True,
        ret_path: bool = False,
        ):
    """Repack/recreate an .rtdc file, optionally stripping the logs

    Parameters
    ----------
    path_in: str or pathlib.Path
        file to compress
    path_out: str or pathlib
        output file path
    strip_basins: bool
        do not write basin information to the output file
    strip_logs: bool
        do not write logs to the output file
    check_suffix: bool
        check suffixes for input and output files
    ret_path: bool
        whether to return the output path

    Returns
    -------
    path_out: pathlib.Path
        output path (with possibly corrected suffix)
    """
    if path_in is None and path_out is None:
        parser = repack_parser()
        args = parser.parse_args()
        path_in = args.input
        path_out = args.output
        strip_basins = args.strip_basins
        strip_logs = args.strip_logs

    allowed_input_suffixes = [".rtdc"]
    if not check_suffix:
        allowed_input_suffixes.append(pathlib.Path(path_in).suffix)

    path_in, path_out, path_temp = common.setup_task_paths(
        path_in, path_out, allowed_input_suffixes=allowed_input_suffixes)

    with h5py.File(path_in) as h5, h5py.File(path_temp, "w") as hc:
        rtdc_copy(src_h5file=h5,
                  dst_h5file=hc,
                  features="all",
                  include_basins=not strip_basins,
                  include_logs=not strip_logs,
                  include_tables=True,
                  meta_prefix="")

    # Finally, rename temp to out
    path_temp.rename(path_out)
    if ret_path:
        return path_out


def repack_parser():
    descr = "Repack an .rtdc file. The difference to dclab-compress " \
            + "is that no logs are added. Other logs can optionally be " \
            + "stripped away. Repacking also gets rid of old clutter " \
            + "data (e.g. previous metadata stored in the HDF5 file)."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('input', metavar="INPUT", type=str,
                        help='Input path (.rtdc file)')
    parser.add_argument('output',  metavar="OUTPUT", type=str,
                        help='Output path (.rtdc file)')
    parser.add_argument('--strip-basins',
                        dest='strip_basins',
                        action='store_true',
                        help='Do not copy any basin information to the '
                             'output file.')
    parser.set_defaults(strip_basins=False)
    parser.add_argument('--strip-logs',
                        dest='strip_logs',
                        action='store_true',
                        help='Do not copy any logs to the output file.')
    parser.set_defaults(strip_logs=False)
    parser.add_argument('--version', action='version',
                        version=f'dclab-repack {version}')
    return parser
