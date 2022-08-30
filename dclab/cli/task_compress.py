"""Compress .rtdc files"""
import argparse
import pathlib
import shutil
import warnings

import hdf5plugin

from ..rtdc_dataset import new_dataset, RTDCWriter
from ..rtdc_dataset.check import IntegrityChecker

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

    if not force:
        # Check whether the input file is already compressed
        # (This is not done in force-mode)
        ic = IntegrityChecker(path_in)
        cue = ic.check_compression()[0]
        if cue.data["uncompressed"] == 0:
            # we are done here
            shutil.copy2(path_in, path_temp)  # copy with metadata
            path_temp.rename(path_out)
            return

    logs = {}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with new_dataset(path_in) as ds:
            ds.export.hdf5(path=path_temp,
                           features=ds.features_innate,
                           filtered=False,
                           compression_kwargs=cmp_kw,
                           override=True)
            logs.update(ds.logs)

        # command log
        logs["dclab-compress"] = common.get_command_log(paths=[path_in])

        # warnings log
        if w:
            logs["dclab-compress-warnings"] = common.assemble_warnings(w)

    # Write log file
    with RTDCWriter(path_temp, compression_kwargs=cmp_kw) as hw:
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
                        help='Force compression, even if the input dataset '
                             + 'is already compressed.')
    parser.set_defaults(force=False)
    return parser
