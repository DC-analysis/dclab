"""command line interface"""
import argparse
import warnings

from ..rtdc_dataset import new_dataset, write_hdf5

from . import common


def condense(path_out=None, path_in=None):
    """Create a new dataset with all (ancillary) scalar-only features"""
    if path_out is None or path_in is None:
        parser = condense_parser()
        args = parser.parse_args()
        path_in = args.input
        path_out = args.output

    path_in, path_out, path_temp = common.setup_task_paths(
        path_in, path_out, allowed_input_suffixes=[".rtdc", ".tdms"])

    logs = {}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with new_dataset(path_in) as ds:
            ds.export.hdf5(path=path_temp,
                           features=ds.features_scalar,
                           filtered=False,
                           compression="gzip",
                           override=True)
            logs.update(ds.logs)

        # command log
        logs["dclab-condense"] = common.get_command_log(paths=[path_in])

        # warnings log
        if w:
            logs["dclab-condense-warnings"] = common.assemble_warnings(w)

    # Write log file
    with write_hdf5.write(path_temp, logs=logs, mode="append",
                          compression="gzip"):
        pass

    # Finally, rename temp to out
    path_temp.rename(path_out)


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
