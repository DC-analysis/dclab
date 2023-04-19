"""Compress .rtdc files"""
import argparse
import pathlib
import warnings

import hdf5plugin
import h5py

from ..rtdc_dataset import rtdc_copy, RTDCWriter
from .. import util
from .._version import version

from . import common


def compress(path_out=None, path_in=None, force=False, check_suffix=True):
    """Create a new dataset with all features compressed losslessly"""
    cmp_kw = hdf5plugin.Zstd(clevel=5)
    if path_out is None or path_in is None:
        parser = compress_parser()
        args = parser.parse_args()
        path_in = args.input
        path_out = args.output
        force = args.force

    allowed_input_suffixes = [".rtdc"]
    if not check_suffix:
        allowed_input_suffixes.append(pathlib.Path(path_in).suffix)

    path_in, path_out, path_temp = common.setup_task_paths(
        path_in, path_out, allowed_input_suffixes=allowed_input_suffixes)

    if force:
        warnings.warn(
            "The `force` keyword argument is deprecated since dclab 0.49.0, "
            "because compressed HDF5 Datasets are now copied and there "
            "is no reason to avoid or use force anymore.",
            DeprecationWarning)

    # command log
    logs = {"dclab-compress": common.get_command_log(paths=[path_in])}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with h5py.File(path_in) as h5, h5py.File(path_temp, "w") as hc:
            rtdc_copy(src_h5file=h5,
                      dst_h5file=hc,
                      features="all",
                      include_logs=True,
                      include_tables=True,
                      meta_prefix="",
                      )

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
        for name in logs:
            hw.store_log(name, logs[name])

    # Finally, rename temp to out
    path_temp.rename(path_out)


def compress_parser():
    descr = "Create a compressed version of an .rtdc file. This can be " \
            + "used for saving disk space (loss-less compression). The " \
            + "data generated during an experiment is usually not compressed."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('input', metavar="INPUT", type=str,
                        help='Input path (.rtdc file)')
    parser.add_argument('output', metavar="OUTPUT", type=str,
                        help='Output path (.rtdc file)')
    parser.add_argument('--force',
                        dest='force',
                        action='store_true',
                        help='DEPRECATED')
    parser.set_defaults(force=False)
    parser.add_argument('--version', action='version',
                        version=f'dclab-compress {version}')
    return parser
