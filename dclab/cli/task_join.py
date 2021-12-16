"""Concatenate .rtdc files"""
import argparse
import time
import warnings

import numpy as np

from ..rtdc_dataset import new_dataset, RTDCWriter
from .. import definitions as dfn

from . import common


class FeatureSetNotIdenticalJoinWarning(UserWarning):
    pass


def join(path_out=None, paths_in=None, metadata=None):
    """Join multiple RT-DC measurements into a single .rtdc file"""
    if metadata is None:
        metadata = {"experiment": {"run index": 1}}
    if path_out is None or paths_in is None:
        parser = join_parser()
        args = parser.parse_args()
        paths_in = args.input
        path_out = args.output

    if len(paths_in) < 2:
        raise ValueError("At least two input files must be specified!")

    paths_in, path_out, path_temp = common.setup_task_paths(
        paths_in, path_out, allowed_input_suffixes=[".rtdc", ".tdms"])

    # Order input files by date
    key_paths = []
    for pp in paths_in:
        with new_dataset(pp) as ds:
            # sorting key
            key = "_".join([ds.config["experiment"]["date"],
                            ds.config["experiment"]["time"],
                            str(ds.config["experiment"]["run index"])
                            ])
            key_paths.append((key, pp))
    sorted_paths = [p[1] for p in sorted(key_paths, key=lambda x: x[0])]

    # Determine temporal offsets
    toffsets = np.zeros(len(sorted_paths), dtype=float)
    for ii, pp in enumerate(sorted_paths):
        with new_dataset(pp) as ds:
            etime = ds.config["experiment"]["time"]
            st = time.strptime(ds.config["experiment"]["date"]
                               + etime[:8],
                               "%Y-%m-%d%H:%M:%S")
            toffsets[ii] = time.mktime(st)
            if len(etime) > 8:
                # floating point time stored as well (HH:MM:SS.SS)
                toffsets[ii] += float(etime[8:])
    toffsets -= toffsets[0]

    logs = {}
    # Determine features to export (based on first file)
    with warnings.catch_warnings(record=True) as w:
        # Catch all FeatureSetNotIdenticalJoinWarnings
        warnings.simplefilter("ignore")
        warnings.simplefilter("always",
                              category=FeatureSetNotIdenticalJoinWarning)
        features = None
        for pp in sorted_paths:
            with new_dataset(pp) as ds:
                # features present
                if features is None:
                    # The initial features are the innate features of the
                    # first file (sorted by time). If we didn't use the innate
                    # features, then the resulting file might become large
                    # (e.g. if we included ancillary features).
                    features = sorted(ds.features_innate)
                else:
                    # Remove features from the feature list, if it is not in
                    # this dataset, or cannot be computed on-the-fly.
                    for feat in features:
                        if feat not in ds.features:
                            features.remove(feat)
                            warnings.warn(
                                f"Excluding feature '{feat}', because "
                                + f"it is not present in '{pp}'!",
                                FeatureSetNotIdenticalJoinWarning)
                    # Warn the user if this dataset has an innate feature that
                    # is being ignored, because it is not an innate feature of
                    # the first dataset.
                    for feat in ds.features_innate:
                        if feat not in features:
                            warnings.warn(
                                f"Ignoring feature '{feat}' in '{pp}', "
                                + "because it is not present in the "
                                + "other files being joined!",
                                FeatureSetNotIdenticalJoinWarning)
        if w:
            logs["dclab-join-feature-warnings"] = common.assemble_warnings(w)

    # Create initial output file
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with new_dataset(sorted_paths[0]) as ds0:
            ds0.export.hdf5(path=path_temp,
                            features=features,
                            filtered=False,
                            override=True,
                            compression="gzip")
        if w:
            logs["dclab-join-warnings-#1"] = common.assemble_warnings(w)

    with RTDCWriter(path_temp) as hw:
        ii = 1
        # Append data from other files
        for pi, ti in zip(sorted_paths[1:], toffsets[1:]):
            ii += 1  # we start with the second dataset
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with new_dataset(pi) as dsi:
                    for feat in features:
                        if feat == "time":
                            # handle time offset
                            fdata = dsi["time"] + ti
                        elif feat == "frame":
                            # handle frame offset
                            fr = dsi.config["imaging"]["frame rate"]
                            frame_offset = ti * fr
                            fdata = dsi["frame"] + frame_offset
                        elif feat == "index_online":
                            if "events/index_online" in hw.h5file:
                                # index_online is usually larger than index
                                ido0 = hw.h5file["events/index_online"][-1] + 1
                            else:
                                ido0 = 0
                            fdata = dsi["index_online"] + ido0
                        else:
                            fdata = dsi[feat]
                        hw.store_feature(feat=feat, data=fdata)
                if w:
                    lkey = f"dclab-join-warnings-#{ii}"
                    logs[lkey] = common.assemble_warnings(w)

        # Logs and configs from source files
        logs["dclab-join"] = common.get_command_log(paths=sorted_paths)
        for ii, pp in enumerate(sorted_paths):
            with new_dataset(pp) as ds:
                # data file logs
                for ll in ds.logs:
                    logs[f"src-#{ii+1}_{ll}"] = ds.logs[ll]
                # configuration
                cfg = ds.config.tostring(sections=dfn.CFG_METADATA).split("\n")
                logs[f"cfg-#{ii+1}"] = cfg

        # Write logs and missing meta data
        for name in logs:
            hw.store_log(name, logs[name])
        hw.store_metadata(metadata)

    # Finally, rename temp to out
    path_temp.rename(path_out)


def join_parser():
    descr = "Join two or more RT-DC measurements. This will produce " \
            + "one larger .rtdc file. The meta data of the dataset " \
            + "that was recorded earliest will be used in the output " \
            + "file. Please only join datasets that were recorded " \
            + "in the same measurement run."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('input', metavar="INPUT", nargs="*", type=str,
                        help='Input paths (.tdms or .rtdc files)')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-o', '--output', metavar="OUTPUT", type=str,
                                help='Output path (.rtdc file)', required=True)
    return parser
