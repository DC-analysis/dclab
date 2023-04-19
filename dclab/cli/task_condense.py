"""Create .rtdc files with scalar-only features"""
import argparse
import pathlib
import warnings

import h5py
import hdf5plugin

from ..rtdc_dataset import new_dataset, rtdc_copy, RTDCWriter
from .. import util
from .._version import version

from . import common


def condense(path_out=None, path_in=None, ancillaries=True,
             check_suffix=True):
    """Create a new dataset with all (ancillary) scalar-only features"""
    cmp_kw = hdf5plugin.Zstd(clevel=5)
    if path_out is None or path_in is None:
        parser = condense_parser()
        args = parser.parse_args()
        path_in = args.input
        path_out = args.output
        ancillaries = not args.no_ancillaries

    allowed_input_suffixes = [".rtdc", ".tdms"]
    if not check_suffix:
        allowed_input_suffixes.append(pathlib.Path(path_in).suffix)

    path_in, path_out, path_temp = common.setup_task_paths(
        path_in, path_out, allowed_input_suffixes=allowed_input_suffixes)

    cmd_dict = {}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with new_dataset(path_in) as ds, h5py.File(path_temp, "w") as hc:
            # If we have an input HDF5 file, then we might readily copy most
            # of the features over using rtdc_copy. If we have a .tdms file,
            # then we have to go the long path.
            if ds.format == "hdf5":
                rtdc_copy(src_h5file=ds.h5file,
                          dst_h5file=hc,
                          features="scalar",
                          include_logs=True,
                          include_tables=True,
                          meta_prefix="")

            hc.require_group("logs")

            # scalar features
            feats_sc = ds.features_scalar
            # loaded (computationally cheap) scalar features
            feats_sc_in = [f for f in ds.features_loaded if f in feats_sc]
            # ancillary features
            feats_sc_anc = list(set(feats_sc) - set(feats_sc_in))

            cmd_dict["features_original_innate"] = ds.features_innate

            if ancillaries:
                features = feats_sc
                cmd_dict["features_computed"] = feats_sc_anc
                if feats_sc_anc:
                    print("Computing ancillary features:",
                          " ".join(feats_sc_anc))
                else:
                    print("No ancillary features to compute.")
            else:
                features = feats_sc_in

            # command log
            logs = {"dclab-condense": common.get_command_log(
                paths=[path_in], custom_dict=cmd_dict)}

            # rename old dclab-condense logs
            for lkey in ["dclab-condense", "dclab-condense-warnings"]:
                if lkey in hc["logs"]:
                    # This is cached, so no worry calling it multiple times.
                    md55m = util.hashfile(path_in, count=80)
                    # rename
                    hc["logs"][f"{lkey}_{md55m}"] = hc["logs"][lkey]
                    del hc["logs"][lkey]

            # Write all remaining scalar features to the file
            # (these are *all* features for .tdms data).
            with RTDCWriter(hc,
                            mode="append",
                            compression_kwargs=cmp_kw,
                            ) as hw:
                for feat in features:
                    if feat not in hc["events"]:
                        hw.store_feature(feat=feat, data=ds[feat])

        # warnings log
        if w:
            logs["dclab-condense-warnings"] = common.assemble_warnings(w)

    # Write log file
    with RTDCWriter(path_temp, compression_kwargs=cmp_kw) as hw:
        for name in logs:
            hw.store_log(name, logs[name])

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
    parser.add_argument('--no-ancillary-features',
                        dest='no_ancillaries',
                        action='store_true',
                        help='Do not compute expensive ancillary features '
                             'such as volume'
                        )
    parser.add_argument('--version', action='version',
                        version=f'dclab-condense {version}')
    return parser
