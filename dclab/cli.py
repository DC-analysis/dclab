"""command line interface"""
import argparse
import pathlib

from .rtdc_dataset import fmt_tdms, load
from . import definitions as dfn


def print_info(string):
    print("\033[1m{}\033[0m".format(string))


def print_alert(string):
    print_info("\033[33m{}".format(string))


def print_violation(string):
    print_info("\033[31m{}".format(string))


def tdms2rtdc():
    """Convert .tdms datasets to the hdf5-based .rtdc file format"""
    parser = tdms2rtdc_parser()
    args = parser.parse_args()

    path_tdms = pathlib.Path(args.tdms_path).resolve()
    path_rtdc = pathlib.Path(args.rtdc_path)

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

        print_info("Converting {:d}/{:d}: {}".format(
            ii + 1, len(files_tdms), ff))
        # load dataset
        ds = load.load_file(ff)
        # create directory
        if not fr.parent.exists():
            fr.parent.mkdir(parents=True)
        # determine features to export
        features = []
        if args.compute_features:
            tocomp = dfn.feature_names
        else:
            tocomp = ds._events
        for feat in tocomp:
            if feat not in dfn.scalar_feature_names:
                if not ds[feat]:
                    # ignore non-existent contour, image, mask, or trace
                    continue
            elif feat not in ds:
                # ignore non-existent feature
                continue
            features.append(feat)
        # export as hdf5
        ds.export.hdf5(path=fr,
                       features=features,
                       filtered=False,
                       override=True)


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
    parser.add_argument('tdms_path', metavar='tdms-path', type=str,
                        help='Input path (tdms file or folder containing '
                             + 'tdms files)')
    parser.add_argument('rtdc_path', metavar='rtdc-path', type=str,
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
            + "implementations of RT-DC recording software (e.g. ShapeIn)."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('path', metavar='path', type=str,
                        help='Path to experimental dataset')
    return parser
