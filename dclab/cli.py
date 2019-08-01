"""command line interface"""
import argparse
import time
import hashlib
import json
import pathlib
import platform
import sys

import h5py
import numpy as np

from .rtdc_dataset import check_dataset, export, fmt_tdms, new_dataset, \
    util, write_hdf5
from . import definitions as dfn
from .compat import PyImportError
from ._version import version

try:
    import imageio
except PyImportError:
    imageio = None

try:
    import nptdms
except PyImportError:
    nptdms = None


def get_command_log(paths, custom_dict={}):
    """Return a json dump of system parameters

    Parameters
    ----------
    paths: list of pathlib.Path or str
        paths of related measurement files; they are hashed
        and included in the "files" key
    custom_dict: dict
        additional user-defined entries; must contain simple
        Python objects (json.dumps must still work)
    """
    data = get_job_info()
    data["files"] = []
    for ii, pp in enumerate(paths):
        fdict = {"name": pathlib.Path(pp).name,
                 "sha256": util.hashfile(pp, hasher_class=hashlib.sha256),
                 "index": ii+1
                 }
        data["files"].append(fdict)
    final_data = {}
    final_data.update(custom_dict)
    final_data.update(data)
    dump = json.dumps(final_data, sort_keys=True, indent=2).split("\n")
    return dump


def get_job_info():
    data = {
        "utc": {
            "date": time.strftime("%Y-%m-%d", time.gmtime()),
            "time": time.strftime("%H:%M:%S", time.gmtime()),
            },
        "system": {
            "info": platform.platform(),
            "machine": platform.machine(),
            "name": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            },
        "python": {
            "implementation": platform.python_implementation(),
            "info": sys.version.replace("\n", ""),
            "version": platform.python_version(),
            },
        "libraries": {
            "dclab": version,
            "h5py": h5py.__version__,
            "numpy": np.__version__,
            }
        }
    if imageio is not None:
        data["libraries"]["imageio"] = imageio.__version__
    if nptdms is not None:
        data["libraries"]["nptdms"] = nptdms.__version__
    return data


def print_info(string):
    print("\033[1m{}\033[0m".format(string))


def print_alert(string):
    print_info("\033[33m{}".format(string))


def print_violation(string):
    print_info("\033[31m{}".format(string))


def condense(path_out=None, path_in=None):
    """Create a new dataset with all (ancillary) scalar-only features"""
    if path_out is None or path_in is None:
        parser = condense_parser()
        args = parser.parse_args()
        path_in = args.input
        path_out = args.output

    path_in = pathlib.Path(path_in)
    path_out = pathlib.Path(path_out)

    if path_out.suffix != ".rtdc":
        path_out = path_out.with_name(path_out.name + ".rtdc")

    logs = {}

    with new_dataset(path_in) as ds:
        features = [f for f in ds.features if f in dfn.scalar_feature_names]
        ds.export.hdf5(path=path_out,
                       features=features,
                       filtered=False,
                       compression="gzip",
                       override=True)
        logs.update(ds.logs)

    # Log
    logs["dclab-condense"] = get_command_log(paths=[path_in])

    # Write log file
    with write_hdf5.write(path_out, logs=logs, mode="append"):
        pass


def condense_parser():
    descr = "Reduce an RT-DC measurement to its scalar-only features " \
            + "(i.e. without `contour`, `image`, `mask`, or `trace`). " \
            + "All available ancillary features are computed."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('input', metavar="INPUT", type=str,
                        help='Input path (.tdms or .rtdc file)')
    parser.add_argument('output', metavar="OUTPUT", type=str,
                        help='Output path (.rtdc file)')
    return parser


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
        ds = new_dataset(pp)
        key = "_".join([ds.config["experiment"]["date"],
                        ds.config["experiment"]["time"],
                        str(ds.config["experiment"]["run index"])
                        ])
        key_paths.append((key, pp))
        del ds
    sorted_paths = [p[1] for p in sorted(key_paths, key=lambda x: x[0])]

    # Create initial output file
    with new_dataset(sorted_paths[0]) as ds0:
        features = sorted(ds0.features_innate)
        ds0.export.hdf5(path=path_out,
                        features=features,
                        filtered=False,
                        compression="gzip")

    with write_hdf5.write(path_out, mode="append") as h5obj:
        # Append data from other files
        for pi in sorted_paths[1:]:
            dsi = new_dataset(pi)
            for feat in features:
                export.hdf5_append(h5obj=h5obj,
                                   rtdc_ds=dsi,
                                   feat=feat,
                                   compression="gzip")
        export.hdf5_autocomplete_config(h5obj)

    # Meta data
    meta = {"experiment": {"run index": 1}}

    # Logs and configs from source files
    logs = {}
    logs["dclab-join"] = get_command_log(paths=sorted_paths)
    for ii, pp in enumerate(sorted_paths):
        with new_dataset(pp) as ds:
            # data file logs
            for ll in ds.logs:
                logs["src-#{}_{}".format(ii+1, ll)] = ds.logs[ll]
            # configuration
            cfg = ds.config.tostring(sections=dfn.CFG_METADATA).split("\n")
            logs["cfg-#{}".format(ii+1)] = cfg

    # Write logs and missing meta data
    with write_hdf5.write(path_out, logs=logs, meta=meta, mode="append"):
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
              skip_initial_empty_image=True, verbose=False):
    """Convert .tdms datasets to the hdf5-based .rtdc file format

    Parameters
    ----------
    path_tdms: str or pathlib.Path
        Path to input .tdms file
    path_rtdc: str or pathlib.Path
        Path to output .rtdc file
    compute_featues: bool
        If `True`, compute all ancillary features and store them in the
        output file
    skip_initial_empty_image: bool
        In old versions of Shape-In, the first image was sometimes
        not stored in the resulting .avi file. In dclab, such images
        are represented as zero-valued images. If `True` (default),
        this first image is not included in the resulting .rtdc file.
    verbose: bool
        If `True`, print messages to stoud
    """
    if path_tdms is None or path_rtdc is None:
        parser = tdms2rtdc_parser()
        args = parser.parse_args()

        path_tdms = pathlib.Path(args.tdms_path).resolve()
        path_rtdc = pathlib.Path(args.rtdc_path)
        compute_features = args.compute_features
        skip_initial_empty_image = not args.include_initial_empty_image
        verbose = True

    if not path_tdms.suffix == ".tdms":
        raise ValueError("Please specify a .tdms file!")

    if not path_rtdc.suffix == ".rtdc":
        path_rtdc = path_rtdc.with_name(path_rtdc.name + ".rtdc")

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
        with new_dataset(ff) as ds:
            # determine features to export
            if compute_features:
                features = ds.features
            else:
                # consider special case for "image", "trace", and "contour"
                # (This will export both "mask" and "contour".
                # The "mask" is computed from "contour" and it is needed
                # by dclab for other ancillary features. We still keep
                # "contour" because it is original data.
                features = ds.features_innate

            if skip_initial_empty_image:
                if ("image" in ds
                        and ds.config["fmt_tdms"]["video frame offset"]):
                    ds.filter.manual[0] = False
                    ds.apply_filter()

            # export as hdf5
            ds.export.hdf5(path=fr,
                           features=features,
                           filtered=True,
                           override=True)

            # write logs
            custom_dict = {}
            # computed features
            cfeats = list(set(features) - set(ds.features_innate))
            if "mask" in features:
                # Mask is always computed from contour data
                cfeats.append("mask")
            custom_dict["ancillary features"] = sorted(cfeats)

            logs = {}
            logs["dclab-tdms2rtdc"] = get_command_log(paths=[ff],
                                                      custom_dict=custom_dict)
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
    parser.add_argument('--include-initial-empty-image',
                        dest='include_initial_empty_image',
                        action='store_true',
                        help='In old versions of Shape-In, the first image '
                             + 'was sometimes not stored in the resulting '
                             + '.avi file. In dclab, such images are '
                             + 'represented as zero-valued images. Set '
                             + 'this option, if you wish to include the '
                             + 'first event with empty image data.')
    parser.set_defaults(include_initial_empty_image=False)
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
    print_info("Checking {}".format(path_in))
    try:
        viol, aler, info = check_dataset(path_in)
    except fmt_tdms.InvalidTDMSFileFormatError:
        print_violation("Invalid tdms file format!")
    except fmt_tdms.IncompleteTDMSFileFormatError:
        print_violation("Incomplete dataset!")
    except fmt_tdms.event_contour.ContourIndexingError:
        print_violation("Invalid contour data!")
    except fmt_tdms.event_image.InvalidVideoFileError:
        print_violation("Invalid image data!")
    else:
        for inf in info:
            print_info(inf)
        for ale in aler:
            print_alert(ale)
        for vio in viol:
            print_violation(vio)
        print_info("Check Complete: {} violations and ".format(len(viol))
                   + "{} alerts".format(len(aler)))


def verify_dataset_parser():
    descr = "Check experimental datasets for completeness. Note that old " \
            + "measurements will most likely fail this verification step. " \
            + "This program is used to enforce data integrity with future " \
            + "implementations of RT-DC recording software (e.g. Shape-In)."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('path', metavar='PATH', type=str,
                        help='Path to experimental dataset')
    return parser
