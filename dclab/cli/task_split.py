"""command line interface"""
import argparse
import pathlib
import warnings

from ..rtdc_dataset import fmt_tdms, new_dataset, write_hdf5

from . import common


def split(path_in=None, path_out=None, split_events=10000,
          skip_initial_empty_image=True, skip_final_empty_image=True,
          ret_out_paths=False, verbose=False):
    """Split a measurement file

    Parameters
    ----------
    path_in: str or pathlib.Path
        Path of input measurement file
    path_out: str or pathlib.Path
        Path to output directory (optional)
    split_events: int
        Maximum number of events in each output file
    skip_initial_empty_image: bool
        Remove the first event of the dataset if the image is zero.
    skip_final_empty_image: bool
        Remove the final event of the dataset if the image is zero.
    ret_out_paths:
        If True, return the list of output file paths.
    verbose: bool
        If `True`, print messages to stdout

    Returns
    -------
    [out_paths]: list of pathlib.Path
        List of generated files (only if `ret_out_paths` is specified)
    """
    if path_in is None:
        parser = split_parser()
        args = parser.parse_args()

        path_in = pathlib.Path(args.path_in).resolve()
        path_out = args.path_out
        split_events = args.split_events
        skip_initial_empty_image = not args.include_empty_boundary_images
        skip_final_empty_image = not args.include_empty_boundary_images
        verbose = True

    if path_out in ["SAME", None]:  # default to input directory
        path_out = path_in.parent

    path_in = pathlib.Path(path_in)
    path_out = pathlib.Path(path_out)
    logs = {"dclab-split": common.get_command_log(paths=[path_in])}
    paths_gen = []
    paths_temp = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # ignore ResourceWarning: unclosed file <_io.BufferedReader...>
        warnings.simplefilter("ignore", ResourceWarning)  # noqa: F821
        if fmt_tdms.NPTDMS_AVAILABLE:  # tdms-related warning filters
            # ignore SlowVideoWarning
            warnings.simplefilter("ignore",
                                  fmt_tdms.event_image.SlowVideoWarning)
            if skip_initial_empty_image:
                # If the initial frame is skipped when empty,
                # suppress any related warning messages.
                warnings.simplefilter(
                    "ignore",
                    fmt_tdms.event_image.InitialFrameMissingWarning)

        with new_dataset(path_in) as ds:
            for ll in ds.logs:
                logs[f"src-{ll}"] = ds.logs[ll]
            num_files = len(ds) // split_events
            if 10 % 4:
                num_files += 1
            for ii in range(num_files):
                pp = path_out / f"{path_in.stem}_{ii+1:04d}.rtdc"
                pt = pp.with_suffix(".rtdc~")
                paths_gen.append(pp)
                paths_temp.append(pt)
                if verbose:
                    print(f"Generating {ii+1:d}/{num_files:d}: {pt}")
                ds.filter.manual[:] = False  # reset filter
                ds.filter.manual[ii*split_events:(ii+1)*split_events] = True
                common.skip_empty_image_events(
                    ds=ds,
                    initial=skip_initial_empty_image,
                    final=skip_final_empty_image)
                ds.apply_filter()
                ds.export.hdf5(
                    path=pt, features=ds.features_innate, filtered=True)

        if w:
            logs["dclab-split-warnings"] = common.assemble_warnings(w)
        sample_name = ds.config["experiment"]["sample"]

    # Add the logs and update sample name
    for ii, pt in enumerate(paths_temp):
        meta = {"experiment": {"sample": f"{sample_name} {ii+1}/{num_files}"}}
        with write_hdf5.write(pt, logs=logs, meta=meta, mode="append",
                              compression="gzip"):
            pass

    for pt, pp in zip(paths_temp, paths_gen):
        pt.rename(pp)

    if ret_out_paths:
        return paths_gen


def split_parser():
    descr = "Split an RT-DC measurement file (.tdms or .rtdc) into multiple " \
            + "smaller .rtdc files."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('path_in', metavar="PATH_IN", type=str,
                        help='Input path (.tdms or .rtdc file)')
    parser.add_argument('--path_out', metavar="PATH_OUT", type=str,
                        default="SAME",
                        help='Output directory (defaults to same directory)')
    parser.add_argument('--split-events', type=int, default=10000,
                        help='Maximum number of events in each output file')
    parser.add_argument('--include-empty-boundary-images',
                        dest='include_empty_boundary_images',
                        action='store_true',
                        help='In old versions of Shape-In, the first or last '
                             + 'images were sometimes not stored in the '
                             + 'resulting .avi file. In dclab, such images '
                             + 'are represented as zero-valued images. Set '
                             + 'this option, if you wish to include these '
                             + 'events with empty image data.')
    parser.set_defaults(include_empty_boundary_images=False)

    return parser
