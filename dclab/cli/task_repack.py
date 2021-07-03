"""command line interface"""
import argparse

import h5py

from ..rtdc_dataset import export, new_dataset, write_hdf5
from .. import definitions as dfn

from . import common


def repack(path_in=None, path_out=None, strip_logs=False):
    """Repack/recreate an .rtdc file, optionally stripping the logs"""
    if path_in is None and path_out is None:
        parser = repack_parser()
        args = parser.parse_args()
        path_in = args.input
        path_out = args.output
        strip_logs = args.strip_logs

    path_in, path_out, path_temp = common.setup_task_paths(
        path_in, path_out, allowed_input_suffixes=[".rtdc"])

    with new_dataset(path_in) as ds, h5py.File(path_temp, "w") as h5:
        # write metadata first (to avoid resetting software version)
        # only export configuration meta data (no user-defined config)
        meta = {}
        for sec in dfn.CFG_METADATA:
            if sec in ds.config:
                meta[sec] = ds.config[sec].copy()

        write_hdf5.write(h5, meta=meta, mode="append")

        if not strip_logs:
            write_hdf5.write(h5, logs=ds.logs, mode="append")

        # write features
        for feat in ds.features_innate:
            export.hdf5_append(h5obj=h5,
                               rtdc_ds=ds,
                               feat=feat,
                               compression="gzip",
                               filtarr=None,
                               time_offset=0)

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
