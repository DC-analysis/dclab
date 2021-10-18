"""Repack (similar to h5repack) .rtdc files"""
import argparse
import pathlib

from ..rtdc_dataset import new_dataset, RTDCWriter
from .. import definitions as dfn

from . import common


def repack(path_in=None, path_out=None, strip_logs=False, check_suffix=True):
    """Repack/recreate an .rtdc file, optionally stripping the logs"""
    if path_in is None and path_out is None:
        parser = repack_parser()
        args = parser.parse_args()
        path_in = args.input
        path_out = args.output
        strip_logs = args.strip_logs

    allowed_input_suffixes = [".rtdc"]
    if not check_suffix:
        allowed_input_suffixes.append(pathlib.Path(path_in).suffix)

    path_in, path_out, path_temp = common.setup_task_paths(
        path_in, path_out, allowed_input_suffixes=allowed_input_suffixes)

    with new_dataset(path_in) as ds, RTDCWriter(path_temp, mode="reset") as hw:
        # write metadata first (to avoid resetting software version)
        # only export configuration meta data (no analysis-related config)
        meta = {}
        for sec in list(dfn.CFG_METADATA.keys()) + ["user"]:
            if sec in ds.config:
                meta[sec] = ds.config[sec].copy()

        hw.store_metadata(meta)

        if not strip_logs:
            for name in ds.logs:
                hw.store_log(name, ds.logs[name])

        # write features
        for feat in ds.features_innate:
            hw.store_feature(feat, ds[feat])

    # Finally, rename temp to out
    path_temp.rename(path_out)


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
    parser.add_argument('--strip-logs',
                        dest='strip_logs',
                        action='store_true',
                        help='Do not copy any logs to the output file.')
    parser.set_defaults(strip_logs=False)
    return parser
