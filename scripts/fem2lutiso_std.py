"""Extract emodulus look-up table (LUT) and isoelastics from simulation data.

See lut_processor for technical details.

Note that a matplotlib window will open when the isoelastics are
created, so you can verify that everything is in order. Just close
that window to proceed.
"""
import argparse
import copy
import json
import pathlib

import matplotlib.pylab as plt
import numpy as np

from lut_recipes import LutProcessor


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

    print("Extracting LUT and isoelastics")
    lup = LutProcessor(path, use_hooks=not raw, verbose=True)
    lut, contours, contour_levels = lup.assemble_lut_and_isoelastics()

    ax = plt.subplot(111, title="Final LUT and isoelastics")
    ax.plot(lut[:, 0], lut[:, 1], ".", color="k")
    for cc in contours:
        ax.plot(cc[:, 0], cc[:, 1])
    plt.show()

    print("Saving LUT and isoelastics")
    save_iso(path.with_name(path.name.rsplit(".", 1)[0] + "_iso.txt"),
             contours, contour_levels, lup.meta)

    save_lut(path.with_name(path.name.rsplit(
        ".", 1)[0] + "_lut.txt"), lut, lup.meta)


def save_iso(path, contours, levels, meta, header=None):
    """Save isoelastics to a text file for usage in dclab"""
    # change identifier to lut identifier
    if header is None:
        header = ["area_um [um^2]", "deform", "emodulus [kPa]"]
    meta = copy.deepcopy(meta)
    if "identifier" in meta:
        idx = meta.pop("identifier")
        meta["lut identifier"] = idx

    lentot = 0
    for cc in contours:
        lentot += len(cc)

    cdat = np.zeros((lentot, 3), dtype=float)
    ii = 0
    for cc, lev in zip(contours, levels):
        sli = slice(ii, ii + len(cc))
        cdat[sli, 0] = cc[:, 0]
        cdat[sli, 1] = cc[:, 1]
        cdat[sli, 2] = lev
        ii += len(cc)

    assert ii == lentot

    save_lut(path, cdat, meta, header=header)


def save_lut(path, lut, meta, header=None):
    """Save LUT to a text file for usage in dclab"""
    if header is None:
        header = ["area_um [um^2]", "deform", "emodulus [kPa]"]
    meta = copy.deepcopy(meta)
    if "column features" in meta:
        # sanity check
        for ii, kk in enumerate(meta["column features"]):
            assert kk in header[ii]
        # remove redundant information; this is stored in the header
        meta.pop("column features")
    with path.open("w") as fd:
        # Metadata
        dump = json.dumps(meta, sort_keys=True, indent=2)
        fd.write("# BEGIN METADATA\r\n")
        for dl in dump.split("\n"):
            fd.write("# " + dl + "\r\n")
        fd.write("# END METADATA\r\n")
        fd.write("#\r\n")

        # Header
        fd.write("# " + "\t".join(header) + "\r\n")

        # Data
        for ii in range(lut.shape[0]):
            line = "{:.5e}\t{:.5e}\t{:.5e}\r\n".format(*lut[ii])
            fd.write(line)


if __name__ == "__main__":
    main()
