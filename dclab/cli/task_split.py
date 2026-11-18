"""Split an .rtdc file into smaller .rtdc files"""
from __future__ import annotations

import argparse
import atexit
import pathlib
import warnings

import hdf5plugin
import numpy as np

from ..rtdc_dataset import fmt_tdms, new_dataset, RTDCWriter
from .._version import version

from . import common


def split(
        path_in: str | pathlib.Path | None = None,
        path_out: str | pathlib.Path | None = None,
        split_events: int = 10000,
        skip_initial_empty_image: bool = True,
        skip_final_empty_image: bool = True,
        ret_out_paths: bool = False,
        verbose: bool = False):
    """Split a measurement file

    Parameters
    ----------
    path_in: str or pathlib.Path
        path of input measurement file
    path_out: str or pathlib.Path
        path to output directory (optional)
    split_events: int
        maximum number of events in each output file
    skip_initial_empty_image: bool
        remove the first event of the dataset if the image is zero
    skip_final_empty_image: bool
        remove the final event of the dataset if the image is zero
    ret_out_paths:
        if True, return the list of output file paths
    verbose: bool
        if True, print messages to stdout

    Returns
    -------
    [out_paths]: list of pathlib.Path
        List of generated files (only if `ret_out_paths` is specified)
    """
    cmp_kw = hdf5plugin.Zstd(clevel=5)
    if path_in is None:
        parser = split_parser()
        args = parser.parse_args()
        path_in = args.path_in
        path_out = args.path_out
        split_events = args.split_events
        skip_initial_empty_image = not args.include_empty_boundary_images
        skip_final_empty_image = not args.include_empty_boundary_images
        verbose = True

    # setup paths
    # input
    assert path_in is not None
    path_in = pathlib.Path(path_in).resolve()
    # output
    if path_out in ["SAME", None]:  # default to input directory
        path_out = path_in.parent
    assert path_out is not None
    path_out = pathlib.Path(path_out)
    if path_out.is_file():
        raise ValueError(f"Output directory '{path_out}' is an existing file")

    logs = {"dclab-split": common.get_command_log(paths=[path_in])}

    paths_gen = []
    paths_temp = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # ignore ResourceWarning: unclosed file <_io.BufferedReader...>
        warnings.simplefilter("ignore", ResourceWarning)
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
            # Internal basin features (e.g. "image_bg") and their mapping
            # features are not exported as regular features, but stored
            # as internal basins in the output files (see below).
            bn_internal = [bn for bn in ds.basins_get_dicts()
                           if bn.get("type") == "internal"]
            if not hasattr(ds, "h5file"):
                # not an HDF5 file (e.g. DCOR), export features as-is
                bn_internal = []
            feats_internal = set()
            for bn in bn_internal:
                feats_internal.update(bn.get("features", []))
                feats_internal.add(bn["mapping"])
            features = [ft for ft in ds.features_innate
                        if ft not in feats_internal]
            num_files = len(ds) // split_events
            if len(ds) % split_events:
                num_files += 1
            for ii in range(num_files):
                pp = path_out / f"{path_in.stem}_{ii+1:04d}.rtdc"
                pt = pp.with_suffix(".rtdc~")
                paths_gen.append(pp)
                paths_temp.append(pt)
                pt.unlink(missing_ok=True)
                atexit.register(pt.unlink, missing_ok=True)
                if verbose:
                    print(f"Generating {ii+1:d}/{num_files:d}: {pt}")
                ds.filter.manual[:] = False  # reset filter
                ds.filter.manual[ii*split_events:(ii+1)*split_events] = True
                common.skip_empty_image_events(
                    ds=ds,
                    initial=skip_initial_empty_image,
                    final=skip_final_empty_image)
                ds.apply_filter()
                ds.export.hdf5(path=pt,
                               features=features,
                               logs=True,
                               tables=True,
                               basins=True,
                               filtered=True,
                               compression_kwargs=cmp_kw,
                               )
                if bn_internal:
                    store_internal_basins(ds=ds,
                                          path=pt,
                                          basin_dicts=bn_internal,
                                          compression_kwargs=cmp_kw)

        if w:
            logs["dclab-split-warnings"] = common.assemble_warnings(w)
        sample_name = ds.config["experiment"]["sample"]

    # Add the logs and update sample name
    for ii, pt in enumerate(paths_temp):
        meta = {"experiment": {"sample": f"{sample_name} {ii+1}/{num_files}"}}
        with RTDCWriter(pt, compression_kwargs=cmp_kw) as hw:
            for name, value in logs.items():
                hw.store_log(name, value)
            hw.store_metadata(meta)

    for pt, pp in zip(paths_temp, paths_gen):
        pt.rename(pp)

    if ret_out_paths:
        return paths_gen


def store_internal_basins(ds, path, basin_dicts, compression_kwargs=None):
    """Store the internal basins of `ds` for its filtered events in `path`

    Only the internal basin data that are required by the filtered events
    are stored, and the mapping (internal basins are always mapped) is
    updated accordingly.
    """
    filter_arr = ds.filter.all
    with RTDCWriter(path,
                    mode="append",
                    compression_kwargs=compression_kwargs) as hw:
        for bn in basin_dicts:
            feats = [ft for ft in bn.get("features", [])
                     if ft in ds.h5file.get("basin_events", {})]
            if not feats:
                continue
            basin_map_orig = ds.h5file["events"][bn["mapping"]][:][filter_arr]
            # Only store the internal basin data that are needed for the
            # events in the output file (sorted unique indices) and map
            # the events to these data.
            indices = np.unique(basin_map_orig)
            basin_map = np.searchsorted(indices, basin_map_orig).astype(
                np.int64)
            internal_data = {ft: ds.h5file["basin_events"][ft][indices]
                             for ft in feats}
            hw.store_basin(basin_name=bn.get("name"),
                           basin_type="internal",
                           basin_format="h5dataset",
                           basin_locs=["basin_events"],
                           basin_descr=bn.get("description"),
                           basin_feats=feats,
                           basin_map=basin_map,
                           internal_data=internal_data,
                           )


def split_parser():
    descr = ("Split an RT-DC measurement file (.tdms or .rtdc) into multiple "
             "smaller .rtdc files.")
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument("path_in", metavar="PATH_IN", type=str,
                        help="Input path (.tdms or .rtdc file)")
    parser.add_argument("--path_out", metavar="PATH_OUT", type=str,
                        default="SAME",
                        help="Output directory (defaults to same directory)")
    parser.add_argument("--split-events", type=int, default=10000,
                        help="Maximum number of events in each output file")
    parser.add_argument("--include-empty-boundary-images",
                        dest="include_empty_boundary_images",
                        action="store_true",
                        help="In old versions of Shape-In, the first or last "
                             "images were sometimes not stored in the "
                             "resulting .avi file. In dclab, such images "
                             "are represented as zero-valued images. Set "
                             "this option, if you wish to include these "
                             "events with empty image data.")
    parser.set_defaults(include_empty_boundary_images=False)
    parser.add_argument("--version", action="version",
                        version=f"dclab-split {version}")
    return parser
