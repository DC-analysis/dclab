"""Create .rtdc files with scalar-only features"""
from __future__ import annotations

import argparse
import pathlib
from typing import List
import warnings

import h5py
import hdf5plugin

from ..rtdc_dataset import (
    fmt_hdf5, new_dataset, rtdc_copy, RTDCWriter, RTDCBase
)
from .. import util
from .._version import version

from . import common


def condense(
        path_in: str | pathlib.Path = None,
        path_out: str | pathlib.Path = None,
        ancillaries: bool = None,
        store_ancillary_features: bool = True,
        store_basin_features: bool = True,
        check_suffix: bool = True,
        ret_path: bool = False
        ):
    """Create a new dataset with all available scalar-only features

    Besides the innate scalar features, this also includes all
    fast-to-compute ancillary and all basin features (`features_loaded`).

    Parameters
    ----------
    path_in: str or pathlib.Path
        file to compress
    path_out: str or pathlib
        output file path
    ancillaries: bool
        DEPRECATED, use `store_ancillary_features` instead
    store_ancillary_features: bool
        compute and store ancillary features in the output file
    store_basin_features: bool
        copy basin features from the input path to the output file;
        Note that the basin information (including any internal
        basin dataset) are always copied over to the new dataset.
    check_suffix: bool
        check suffixes for input and output files
    ret_path: bool
        whether to return the output path

    Returns
    -------
    path_out: pathlib.Path (optional)
        output path (with possibly corrected suffix)
    """
    if ancillaries is not None:
        warnings.warn("Please use `store_ancillary_features` instead of "
                      "`ancillaries`", DeprecationWarning)
        store_ancillary_features = ancillaries

    if path_out is None or path_in is None:
        parser = condense_parser()
        args = parser.parse_args()
        path_in = args.input
        path_out = args.output
        store_ancillary_features = not args.no_ancillaries
        store_basin_features = not args.no_basins

    allowed_input_suffixes = [".rtdc", ".tdms"]
    if not check_suffix:
        allowed_input_suffixes.append(pathlib.Path(path_in).suffix)

    path_in, path_out, path_temp = common.setup_task_paths(
        path_in, path_out, allowed_input_suffixes=allowed_input_suffixes)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # We use `store_basin_features` during initialization (to avoid
        # conflicts with ancillary features) and in the actual function
        # as well, to correctly determine which features to use.
        with new_dataset(path_in, enable_basins=store_basin_features) as ds, \
                h5py.File(path_temp, "w") as h5_cond:
            condense_dataset(ds=ds,
                             h5_cond=h5_cond,
                             store_ancillary_features=store_ancillary_features,
                             store_basin_features=store_basin_features,
                             warnings_list=w)

    # Finally, rename temp to out
    path_temp.rename(path_out)
    if ret_path:
        return path_out


def condense_dataset(
        ds: RTDCBase,
        h5_cond: h5py.File,
        ancillaries: bool = None,
        store_ancillary_features: bool = True,
        store_basin_features: bool = True,
        warnings_list: List = None):
    """Condense a dataset using low-level HDF5 methods

    For ancillary and basin features, high-level dclab methods are used.
    """
    if ancillaries is not None:
        warnings.warn("Please use `store_ancillary_features` instead of "
                      "`ancillaries`", DeprecationWarning)
        store_ancillary_features = ancillaries

    cmp_kw = hdf5plugin.Zstd(clevel=5)
    cmd_dict = {}

    # If we have an input HDF5 file, then we might readily copy most
    # of the features over using rtdc_copy. If we have a .tdms file,
    # then we have to go the long route.
    if isinstance(ds, fmt_hdf5.RTDC_HDF5):
        rtdc_copy(src_h5file=ds.h5file,
                  dst_h5file=h5_cond,
                  features="scalar",
                  include_basins=True,
                  include_logs=True,
                  include_tables=True,
                  meta_prefix="")

    h5_cond.require_group("logs")

    # scalar features
    feats_sc = ds.features_scalar
    # loaded (computationally cheap) scalar features
    feats_sc_loaded = [f for f in ds.features_loaded if f in feats_sc]
    # internal basin features that have already been copied with `rtdc_copy`
    feats_sc_basint = sorted(h5_cond.get("basin_events", {}).keys())
    # features that are excluded, because we already copied them
    feats_exclude = feats_sc_loaded + feats_sc_basint

    cmd_dict["features_original_innate"] = ds.features_innate

    features = set(feats_sc_loaded)
    if store_basin_features:
        feats_sc_basin = [f for f in ds.features_basin if
                          (f in feats_sc and f not in feats_exclude)]
        cmd_dict["features_basin"] = feats_sc_basin
        if feats_sc_basin:
            print(f"Using basin features {feats_sc_basin}")
            features |= set(feats_sc_basin)

    if store_ancillary_features:
        feats_sc_anc = [f for f in ds.features_ancillary if
                        (f in feats_sc and f not in feats_exclude)]
        cmd_dict["features_ancillary"] = feats_sc_anc
        if feats_sc_anc:
            features |= set(feats_sc_anc)
            print(f"Using ancillary features {feats_sc_anc}")

    # command log
    logs = {"dclab-condense": common.get_command_log(
        paths=[ds.path], custom_dict=cmd_dict)}

    # rename old dclab-condense logs
    for l_key in ["dclab-condense", "dclab-condense-warnings"]:
        if l_key in h5_cond["logs"]:
            # This is cached, so no worry calling it multiple times.
            md5_cfg = util.hashobj(ds.config)
            # rename
            new_log_name = f"{l_key}_{md5_cfg}"
            if new_log_name not in h5_cond["logs"]:
                # If the user repeatedly condensed one file, then there is
                # no benefit in storing the log under a different name (the
                # metadata did not change). Only write the log if it does
                # not already exist.
                h5_cond["logs"][f"{l_key}_{md5_cfg}"] = h5_cond["logs"][l_key]
            del h5_cond["logs"][l_key]

    with RTDCWriter(h5_cond,
                    mode="append",
                    compression_kwargs=cmp_kw,
                    ) as hw:
        # Write all remaining scalar features to the file
        # (these are *all* scalar features in the case of .tdms data).
        for feat in features:
            if feat not in h5_cond["events"]:
                hw.store_feature(feat=feat, data=ds[feat])

        # collect warnings log
        if warnings_list:
            logs["dclab-condense-warnings"] = \
                common.assemble_warnings(warnings_list)

        # Write logs
        for name in logs:
            hw.store_log(name, logs[name])


def condense_parser():
    descr = "Reduce an RT-DC measurement to its scalar-only features " \
            + "(i.e. without `contour`, `image`, `mask`, or `trace`). " \
            + "All available ancillary features are computed."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('input', metavar="INPUT", type=str,
                        help='Input path (.tdms or .rtdc file)')
    parser.add_argument('output', metavar="OUTPUT", type=str,
                        help='Output path (.rtdc file)')
    parser.add_argument('--no-ancillary-features',
                        dest='no_ancillaries',
                        action='store_true',
                        help='Do not compute expensive ancillary features '
                             'such as volume'
                        )
    parser.set_defaults(no_ancillaries=False)
    parser.add_argument('--no-basin-features',
                        dest='no_basins',
                        action='store_true',
                        help='Do not store basin-based feature data from the '
                             'input file in the output file'
                        )
    parser.set_defaults(no_basins=False)
    parser.add_argument('--version', action='version',
                        version=f'dclab-condense {version}')
    return parser
