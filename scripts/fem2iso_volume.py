"""Extract volume-deformation isoelasticity lines from simulation data

The simulation dataset must be an HDF5 file with a specific structure
that contains the results of FEM simulations. The original HDF5 files
were provided by Lucas Wittwer.

Creating volume-deformation isoelasticity lines means that a full LUT
is generated from which the isoelasticity lines are then interpolated.
Since the LUT has no function in dclab, it is not exported.

Additional post-processing hooks for LUT or isoelastics generation
are defined in the Python files named according to the LUT identifier
in the "fem_hooks" subdirectory.

The discussion related to this script is archived in issue #70 (dclab).

An example HDF5 file can be found on figshare
(LE-2D-FEM-19, https://doi.org/10.6084/m9.figshare.12155064.v4).
"""
import argparse
import pathlib

import numpy as np

import fem2lutiso_std


def get_lut_volume(path, processing=True):
    """Extract the volume LUT from an HDF5 file provided by Lucas Wittwer

    Parameters
    ----------
    path: str or pathlib.Path
        Path to an hdf5 file
    processing: bool
        whether or not to perform post-processing;
        Post-processing is identifier based - you may create a
        Python file named after the LUT identifier and define the
        function `process_lut_volume_deform` therein (see the
        "fem_hooks" subdirectory for examples)
    """
    lut_base, meta = fem2lutiso_std.get_lut_base(path)
    lut = np.zeros((len(lut_base["emodulus"]), 3), dtype=float)
    lut[:, 0] = lut_base["volume"]
    lut[:, 1] = lut_base["deform"]
    lut[:, 2] = lut_base["emodulus"]

    if processing:
        phook = fem2lutiso_std.get_processing_hook(meta["identifier"],
                                                   "process_lut_volume_deform")
        if phook is not None:
            lut = phook(lut)

    meta["column features"] = ["volume", "deform", "emodulus"]

    return lut, meta


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input', metavar="INPUT", type=str,
                        help='Input path (.hdf5 file)')
    parser.add_argument("--raw",
                        dest='raw',
                        action='store_true',
                        help="do not perform data post-processing",
                        )
    parser.set_defaults(raw=False)

    args = parser.parse_args()
    path = pathlib.Path(args.input)
    raw = args.raw

    if raw:
        print("Skipping all post-processing steps!")

    print("Extracting volume-deformation LUT")
    lut, meta = get_lut_volume(path, processing=not raw)

    print("Extracting volume-deformation isoelastics")
    contours, levels = fem2lutiso_std.get_isoelastics(lut, meta,
                                                      processing=not raw)
    fem2lutiso_std.save_iso(
        path=path.with_name(path.name.rsplit(".", 1)[0] + "_volume_iso.txt"),
        contours=contours,
        levels=levels,
        meta=meta,
        header=["volume [um^3]", "deform", "emodulus [kPa]"])


if __name__ == "__main__":
    main()
