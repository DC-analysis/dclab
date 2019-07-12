"""command line interface"""
import argparse
import datetime
import hashlib
import json
import pathlib
import sys

import h5py
import numpy as np

from .rtdc_dataset import export, fmt_tdms, load, util, write_hdf5
from . import definitions as dfn

from ._version import version


def get_command_log(paths):
    data = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "platform": sys.platform,
        "python": sys.version.replace("\n", ""),
        "dclab": version,
        "h5py": h5py.__version__,
        "numpy": np.__version__,
        "files": [],
    }
    for ii, pp in enumerate(paths):
        fdict = {"name": pathlib.Path(pp).name,
                 "sha256": util.hashfile(pp, hasher_class=hashlib.sha256),
                 "index": ii+1
                 }
        data["files"].append(fdict)
    dump = json.dumps(data, sort_keys=True, indent=2).split("\n")
    return dump


def print_info(string):
    print("\033[1m{}\033[0m".format(string))


def print_alert(string):
    print_info("\033[33m{}".format(string))


def print_violation(string):
    print_info("\033[31m{}".format(string))


def join(path_out=None, paths_in=None):
    """Join multiple RT-DC measurements into a single .rtdc file"""
    if path_out is None or paths_in is None:
        parser = join_parser()
        args = parser.parse_args()
        paths_in = args.input
        path_out = args.output

    if len(paths_in) < 2:
        raise ValueError("At least two input files must be specified!")

    if pathlib.Path(path_out).exists():
        raise ValueError("Output file '{}' already exists!".format(path_out))

    # Determine input file order
    key_paths = []
    for pp in paths_in:
        ds = load.new_dataset(pp)
        key = "_".join([ds.config["experiment"]["date"],
                        ds.config["experiment"]["time"],
                        str(ds.config["experiment"]["run index"])
                        ])
        key_paths.append((key, pp))
        del ds
    sorted_paths = [p[1] for p in sorted(key_paths, key=lambda x: x[0])]

    # Create initial output file
    ds0 = load.new_dataset(sorted_paths[0])
    # Check for features existence (fmt_tdms may have empty "image", ...)
    features = sorted([f for f in ds0._events if f in ds0])
    ds0.export.hdf5(path=path_out,
                    features=features,
                    filtered=False,
                    compression="gzip")

    with write_hdf5.write(path_out, mode="append") as h5obj:
        # Append data from other files
        for pi in sorted_paths[1:]:
            dsi = load.new_dataset(pi)
            for feat in features:
                export.hdf5_append(h5obj=h5obj,
                                   rtdc_ds=dsi,
                                   feat=feat,
                                   compression="gzip")

    with load.new_dataset(path_out) as dsf:
        # Important keyword arguments
        meta = {"experiment": {"event count": len(dsf),
                               "run index": 1}}

    # Logs and configs from source files
    logs = {}
    logs["dclab-join"] = get_command_log(paths=sorted_paths)
    for ii, pp in enumerate(sorted_paths):
        with load.new_dataset(pp) as ds:
            # data file logs
            for ll in ds.logs:
                logs["src-#{}_{}".format(ii+1, ll)] = ds.logs[ll]
            # configuration
            cfg = ds.config.tostring(sections=dfn.CFG_METADATA).split("\n")
            logs["cfg-#{}".format(ii+1)] = cfg

    # Write missing meta data
    with write_hdf5.write(path_out,
                          logs=logs,
                          meta=meta,
                          mode="append") as h5obj:
        pass


def join_parser():
    descr = "Join two or more RT-DC measurements. This will produce " \
            + "one larger .rtdc file. The meta data of the dataset " \
            + "that was recorded earliest will be used in the output " \
            + "file. Please only join datasets that were recorded " \
            + "in the same measurement run."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('input', metavar="INPUT", nargs="*", type=str,
                        help='Input paths (.tdms or .rtdc files)')
    parser.add_argument('-o', '--output', type=str,
                        help='Output path (.rtdc file)')
    return parser


def tdms2rtdc(path_tdms=None, path_rtdc=None, compute_features=False,
              verbose=False):
    """Convert .tdms datasets to the hdf5-based .rtdc file format"""
    if path_tdms is None or path_rtdc is None:
        parser = tdms2rtdc_parser()
        args = parser.parse_args()

        path_tdms = pathlib.Path(args.tdms_path).resolve()
        path_rtdc = pathlib.Path(args.rtdc_path)
        compute_features = args.compute_features
        verbose = True

    # Determine whether input path is a tdms file or a directory
    if path_tdms.is_dir():
        files_tdms = fmt_tdms.get_tdms_files(path_tdms)
        if path_rtdc.is_file():
            raise ValueError("rtdc_path is a file: {}".format(path_rtdc))
        files_rtdc = []
        for ff in files_tdms:
            ff = pathlib.Path(ff)
            rp = ff.relative_to(path_tdms)
            # determine output file name (same relative path)
            rpr = path_rtdc / rp.with_suffix(".rtdc")
            files_rtdc.append(rpr)
    else:
        files_tdms = [path_tdms]
        files_rtdc = [path_rtdc]

    for ii in range(len(files_tdms)):
        ff = pathlib.Path(files_tdms[ii])
        fr = pathlib.Path(files_rtdc[ii])

        if verbose:
            print_info("Converting {:d}/{:d}: {}".format(
                ii + 1, len(files_tdms), ff))
        # create directory
        if not fr.parent.exists():
            fr.parent.mkdir(parents=True)
        # load and export dataset
        with load.new_dataset(ff) as ds:
            # determine features to export
            if compute_features:
                features = ds.features
            else:
                # consider special case for "image", "trace", and "contour"
                # (This will export both "mask" and "contour".
                # The "mask" is computed from "contour" and it is needed
                # by dclab for other ancillary features. We still keep
                # "contour" because it is original data.
                features = [f for f in ds._events if f in ds]

            # export as hdf5
            ds.export.hdf5(path=fr,
                           features=features,
                           filtered=False,
                           override=True)

            # write logs
            logs = {}
            logs["dclab-tdms2rtdc"] = get_command_log(paths=[ff])
            logs.update(ds.logs)
            with write_hdf5.write(fr, logs=logs, mode="append"):
                pass


def tdms2rtdc_parser():
    descr = "Convert RT-DC .tdms files to the hdf5-based .rtdc file format. " \
            + "Note: Do not delete original .tdms files after conversion. " \
            + "The conversion might be incomplete."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('--compute-ancillary-features',
                        dest='compute_features',
                        action='store_true',
                        help='Compute features, such as volume or emodulus, '
                             + 'that are otherwise computed on-the-fly. '
                             + 'Use this if you want to minimize analysis '
                             + 'time in e.g. Shape-Out. CAUTION: ancillary '
                             + 'feature recipes might be subject to change '
                             + '(e.g. if an error is found in the recipe). '
                             + 'Disabling this option maximizes '
                             + 'compatibility with future versions and '
                             + 'allows to isolate the original data.')
    parser.set_defaults(compute_features=False)
    parser.add_argument('tdms_path', metavar="TDMS_PATH", type=str,
                        help='Input path (tdms file or folder containing '
                             + 'tdms files)')
    parser.add_argument('rtdc_path', metavar="RTDC_PATH", type=str,
                        help='Output path (file or folder), existing data '
                             + 'will be overridden')
    return parser


def verify_dataset():
    """Perform checks on experimental datasets"""
    parser = verify_dataset_parser()
    args = parser.parse_args()
    path_in = pathlib.Path(args.path).resolve()
    viol, aler, info = load.check_dataset(path_in)
    print_info("Checking {}".format(path_in))
    for inf in info:
        print_info(inf)
    for ale in aler:
        print_alert(ale)
    for vio in viol:
        print_violation(vio)
    print_info("Check Complete: {} violations and {} alerts".format(len(viol),
                                                                    len(aler)))


def verify_dataset_parser():
    descr = "Check experimental datasets for completeness. Note that old " \
            + "measurements will most likely fail this verification step. " \
            + "This program is used to enforce data integrity with future " \
            + "implementations of RT-DC recording software (e.g. Shape-In)."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('path', metavar='PATH', type=str,
                        help='Path to experimental dataset')
    return parser
