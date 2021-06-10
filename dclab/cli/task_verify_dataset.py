"""command line interface"""
import argparse
import pathlib
import sys

from ..rtdc_dataset import check_dataset, fmt_tdms

from . import common


def verify_dataset(path_in=None):
    """Perform checks on experimental datasets"""
    if path_in is None:
        parser = verify_dataset_parser()
        args = parser.parse_args()
        path_in = pathlib.Path(args.path).resolve()
    common.print_info(f"Checking {path_in}")
    # The exit status of this script. Non-zero exit status means:
    # 1: alerts
    # 2: violations
    # 3: alerts and violations
    # 4: other error
    exit_status = 4
    try:
        viol, aler, info = check_dataset(path_in)
    except fmt_tdms.InvalidTDMSFileFormatError:
        common.print_violation("Invalid tdms file format!")
    except fmt_tdms.IncompleteTDMSFileFormatError:
        common.print_violation("Incomplete dataset!")
    except fmt_tdms.ContourIndexingError:
        common.print_violation("Invalid contour data!")
    except fmt_tdms.InvalidVideoFileError:
        common.print_violation("Invalid image data!")
    except BaseException as e:
        common.print_violation(f"{e.__class__.__name__}: {', '.join(e.args)}")
    else:
        for inf in info:
            common.print_info(inf)
        for ale in aler:
            common.print_alert(ale)
        for vio in viol:
            common.print_violation(vio)
        common.print_info(f"Check Complete: {len(viol)} violations and "
                          + f"{len(aler)} alerts")
        if aler and viol:
            exit_status = 3
        elif aler:
            exit_status = 1
        elif viol:
            exit_status = 2
        else:
            # everything is ok
            exit_status = 0
    finally:
        # return sys.exit for testing (monkeypatched)
        return sys.exit(exit_status)


def verify_dataset_parser():
    descr = "Check experimental datasets for completeness. This command " \
            + "is used e.g. to enforce data integrity with Shape-In. The " \
            + "following exit codes are defined: ``0: valid dataset``, " \
            + "``1: alerts encountered``, ``2: violations encountered``, " \
            + "``3: alerts and violations``, ``4: other error``."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('path', metavar='PATH', type=str,
                        help='Path to experimental dataset')
    return parser
