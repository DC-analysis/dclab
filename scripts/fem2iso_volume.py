"""Extract volume-deformation isoelasticity lines from simulation data

The simulation dataset must be an HDF5 file with a specific structure
that contains the results of FEM simulations. The original HDF5 files
were provided by Lucas Wittwer.

Creating volume-deformation isoelasticity lines means that a full LUT
is generated from which the isoelasticity lines are then interpolated.
Since the LUT has no function in dclab, it is not exported.

Additional post-processing hooks for LUT or isoelastics generation
are defined in the Python files named according to the LUT identifier
in the "lut_recipes" subdirectory.

The discussion related to this script is archived in issue #70 (dclab).

An example HDF5 file can be found on figshare
(LE-2D-FEM-19, https://doi.org/10.6084/m9.figshare.12155064.v4).
"""
import argparse
import pathlib

import matplotlib.pylab as plt

from lut_recipes import LutProcessor
import fem2lutiso_std


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
    lup = LutProcessor(path, use_hooks=not raw, featx="volume", verbose=True)
    lut, contours, levels = lup.assemble_lut_and_isoelastics()

    ax = plt.subplot(111, title="Final LUT and isoelastics")
    ax.plot(lut[:, 0], lut[:, 1], ".", color="k")
    for cc in contours:
        ax.plot(cc[:, 0], cc[:, 1])
    plt.show()

    fem2lutiso_std.save_iso(
        path=path.with_name(path.name.rsplit(".", 1)[0] + "_volume_iso.txt"),
        contours=contours,
        levels=levels,
        meta=lup.meta,
        header=["volume [um^3]", "deform", "emodulus [kPa]"])


if __name__ == "__main__":
    main()
