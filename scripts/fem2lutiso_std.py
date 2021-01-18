"""Extract emodulus look-up table (LUT) and isoelastics from simulation data.

The simulation dataset must be an HDF5 file with a specific structure
that contains the results of FEM simulations. The original HDF5 files
were provided by Lucas Wittwer. Any LUT added to dclab after version
0.23.0 was extracted and created using this script.

The following data post-processing is performed for the LUT (depending
on the HDF5 root attributes):
- If the LUT "model" is "linear elastic" and the LUT "dimensionality"
  is "2Daxis", then the LUT is complemented with analytical values
  from "LUT_analytical_linear-elastic_2Daxis.txt" for small deformation
  and area below 200um. The original FEM simulations did not cover that
  area, because of discretization problems (deformations smaller than
  0.005 could not be resolved).
- If the LUT "dimensionality" is "2Daxis" (rotationally symmetric
  simulation), then the LUT is cropped at a maximum area of 290um^2.
  The reason is that the axis-symmetric model becomes inaccurate when
  the object boundary comes close to the channel walls (the actual
  flow profile in a rectangular cross-section channel is not anymore
  rotationally symmetric around the object). In addition, there have
  been numerical errors due to meshing if the area is above 290um^2.
- Redundant values in the LUT are removed by checking whether they
  can be reproduced through linear interpolation of the remaining
  values.

The following data post-processing is performed for the isoelastics:
- If the LUT "model" is "linear elastic" and the LUT "dimensionality"
  is "2Daxis", isoelastics with a maximum deformation above 0.159
  are cropped by a few points (depends on resolution of grid
  interpolation).

Note that a matplotlib window will open when the isoelastics are
created so you can verify that everything is in order. Just close
that window to proceed.

An example HDF5 file can be found on figshare
(https://doi.org/10.6084/m9.figshare.12155064.v2).
"""
import argparse
import copy
import json
import numbers
import pathlib


from dclab.features import emodulus
from dclab.external import skimage
import h5py
import matplotlib.pylab as plt
import numpy as np
import scipy.interpolate as spint


def get_isoelastics(lut, meta, processing=True):
    """Compute equidistant isoelastics from a LUT

    The LUT is interpolated on a 200x200 pixel grid and then
    isoealstics are determined by finding contour lines.

    Notes
    -----
    If the LUT "model" is "linear elastic" and the LUT "dimensionality"
    is "2Daxis", isoelastics with a maximum deformation above 0.159
    are cropped by a few points (depends on resolution of grid
    interpolation).
    """
    wlut = np.array(lut, copy=True)
    # normalize
    area_norm = wlut[:, 0].max()
    emodulus.normalize(wlut[:, 0], area_norm)

    defo_norm = wlut[:, 1].max()
    emodulus.normalize(wlut[:, 1], defo_norm)

    # Compute gridded version
    size = 200  # please don't change (check contour downsampling)
    area_min = wlut[:, 0].min()
    deform_min = wlut[:, 1].min()
    x = np.linspace(area_min, 1, size, endpoint=True)
    y = np.linspace(deform_min, 1, size, endpoint=True)

    xm, ym = np.meshgrid(x, y, indexing="ij")

    emod = spint.griddata((wlut[:, 0], wlut[:, 1]), wlut[:, 2],
                          (xm, ym), method="linear")

    # Determine the levels via a line plot through the
    # given LUT.
    # These indices are selected like that, because emod[0,:] is usually nan.
    ids = size // 50

    # These are the original levels (by Christoph Herold):
    # levels = [0.9 1.2 1.5 1.8 2.1 2.55 3. 3.6 4.2 5.4 6.9]
    # These are the new levels (by the following algorithm for linear
    # elastic material and 2Daxis dimensionality):
    # levels = [0.93 1.16 1.4 1.67 1.99 2.4 2.93 3.67 4.84 6.94 12.13]
    deform_start = ym[0, :][~np.isnan(emod[ids, :])].max()
    area_end = xm[:, 0][~np.isnan(emod[:, ids])].max()

    xlev = np.linspace(area_min, area_end, 13, endpoint=True)
    ylev = np.linspace(deform_start, deform_min, 13, endpoint=True)

    elev = spint.griddata((wlut[:, 0], wlut[:, 1]), wlut[:, 2],
                          (xlev, ylev), method="linear")
    levels = elev[1:-1]
    levels = np.round(levels, 2)

    contours_px = []
    contours = []
    for level in levels:
        conts = skimage.measure.find_contours(emod,
                                              level=level,
                                              )
        # get the longest one
        idx = np.argmax([len(cc) for cc in conts])
        cc = conts[idx]
        # remove nan values
        cc = cc[~np.isnan(np.sum(cc, axis=1))]
        # downsample contour
        keep = np.zeros(cc.shape[0], dtype=bool)
        keep[0] = True
        keep[-1] = True
        keep[::2] = True
        cc = cc[keep, :]
        contours_px.append(cc)
        # convert pixel to absolute area_um and deform
        ccu = np.zeros_like(cc)
        ccu[:, 0] = (cc[:, 0] / size * (1 - area_min) + area_min) * area_norm
        ccu[:, 1] = (cc[:, 1] / size * (1 - deform_min) +
                     deform_min) * defo_norm
        # avoid points at the boundary of the LUT for large deformations
        if (processing
            and meta["model"] == "linear elastic"
                and meta["dimensionality"] == "2Daxis"):
            if ccu[-1, 1] > 0.159:
                print("...Post-Processing: Cropping isoelastics above "
                      + "deformation 0.159 by {} points.".format(ids))
                ccu = ccu[:-ids, :]
        contours.append(ccu)

    plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(121)
    ax1.imshow(emod)
    for cont in contours_px:
        ax1.plot(cont[:, 1], cont[:, 0], "x")

    ax2 = plt.subplot(122)
    for cont in contours:
        ax2.plot(cont[:, 0], cont[:, 1], "-")

    plt.tight_layout()
    plt.show()

    assert len(contours) == len(levels)
    return contours, levels


def get_lut(path, processing=True):
    """Extract the LUT from a FEM simulation HDF5 file (Lucas Wittwer)

    Notes
    -----
    - If the LUT "model" is "linear elastic" and the LUT "dimensionality"
      is "2Daxis", then the LUT is complemented with analytical values
      from "LUT_analytical_linear-elastic_2Daxis.txt" for small deformation
      and area below 200um. The original FEM simulations did not cover that
      area, because of discretization problems (deformations smaller than
      0.005 could not be resolved).
    - If the LUT "dimensionality" is "2Daxis" (rotationally symmetric
      simulation), then the LUT is cropped at a maximum area of 290um^2.
      The reason is that the axis-symmetric model becomes inaccurate when
      the object boundary comes close to the channel walls (the actual
      flow profile in a rectangular cross-section channel is not anymore
      rotationally symmetric around the object). In addition, there have
      been numerical errors due to meshing if the area is above 290um^2.
    """
    lut_base, meta = get_lut_base(path)
    lutsize = len(lut_base["emodulus"])
    lut = np.zeros((lutsize, 3), dtype=float)
    lut[:, 0] = lut_base["area_um"]
    lut[:, 1] = lut_base["deform"]
    lut[:, 2] = lut_base["emodulus"]

    if processing:
        if (meta["model"] == "linear elastic"
                and meta["dimensionality"] == "2Daxis"):
            ap = "LUT_analytical_linear-elastic_2Daxis.txt"
            print("...Post-Processing: Adding analytical LUT from {}.".format(
                ap))
            # load analytical data
            lut_ana = np.loadtxt(pathlib.Path(__file__).parent / ap)
            lut = np.concatenate((lut, lut_ana))

        if meta["dimensionality"] == "2Daxis":
            print("...Post-Processing: Cropping LUT at 290um^2.")
            lut = lut[lut[:, 0] < 290]

    meta["column features"] = ["area_um", "deform", "emodulus"]

    return lut, meta


def get_lut_base(path):
    """Extract features from a FEM simulation HDF5 file (Lucas Wittwer)

    Returns
    -------
    data: dict
        Each key is a dclab feature, the value is a 1D ndarray
    meta: dict
        Metadata extracted from the HDF5 file
    """
    area_um = []
    deform = []
    emod = []
    volume = []
    with h5py.File(path, "r") as h5:
        assert h5.attrs["flow_rate_unit"].decode("utf-8") == "uL/s"
        assert h5.attrs["channel_width_unit"].decode("utf-8") == "um"
        assert h5.attrs["fluid_viscosity_unit"].decode("utf-8") == "Pa s"
        meta = dict(h5.attrs)
        # convert viscosity to mPa*s
        meta["fluid_viscosity_unit"] = b"mPa s"
        meta["fluid_viscosity"] = meta["fluid_viscosity"] * 1000
        for key in meta:
            if isinstance(meta[key], bytes):
                meta[key] = meta[key].decode("utf-8")
            elif isinstance(meta[key], numbers.Integral):
                meta[key] = int(meta[key])
            else:
                meta[key] = float(meta[key])
        for Ek in h5.keys():
            assert h5[Ek].attrs["emodulus_unit"].decode("utf-8") == "Pa"
            for simk in h5[Ek].keys():
                sim = h5[Ek][simk]
                area_um.append(sim.attrs["area"])
                assert sim.attrs["area_unit"].decode("utf-8") == "um^2"
                deform.append(sim.attrs["deformation"])
                assert sim.attrs["deformation_unit"].decode("utf-8") == ""
                volume.append(sim.attrs["volume"])
                assert sim.attrs["volume_unit"].decode("utf-8") == "um^3"
                emod.append(h5[Ek].attrs["emodulus"]/1000)
    data = {"area_um": np.array(area_um),
            "deform": np.array(deform),
            "emodulus": np.array(emod),
            "volume": np.array(volume),
            }
    return data, meta


def save_iso(path, contours, levels, meta,
             header=["area_um [um^2]", "deform", "emodulus [kPa]"]):
    """Save isoelastics to a text file for usage in dclab"""
    # change identifier to lut identifier
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


def save_lut(path, lut, meta,
             header=["area_um [um^2]", "deform", "emodulus [kPa]"]):
    """Save LUT to a text file for usage in dclab"""
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


def shrink_lut(lut):
    """Remove redundant values in a LUT

    This is achieved by checking whether they can be reproduced
    through linear interpolation from the remaining values.
    """
    wlut = np.array(lut, copy=True)
    # normalize
    area_norm = wlut[:, 0].max()
    emodulus.normalize(wlut[:, 0], area_norm)

    defo_norm = wlut[:, 1].max()
    emodulus.normalize(wlut[:, 1], defo_norm)

    keep = np.ones(wlut.shape[0], dtype=bool)

    checked = np.zeros(wlut.shape[0], dtype=int)

    # Take out a few points and see whether linear interpolation would
    # be enough to recover it.
    for ii in range(wlut.shape[0]):
        if checked[ii] >= 4:
            continue
        ids = np.arange(ii, wlut.shape[0], step=47)
        cur = np.array(keep, copy=True)
        cur[ids] = False
        emod = spint.griddata((wlut[cur, 0], wlut[cur, 1]), wlut[cur, 2],
                              (wlut[ids, 0], wlut[ids, 1]),
                              method='linear')

        for em, im in zip(emod, ids):
            checked[im] += 1
            if np.isnan(em):
                continue
            elif np.allclose(em, wlut[im, 2], atol=1e-6, rtol=1e-5):
                keep[im] = False
                checked[im] = 4
    print("... -> Removed {} out of {} data points.".format(
        np.sum(~keep), keep.size))

    lut_new = 1*lut[keep, :]

    return lut_new


if __name__ == "__main__":
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

    print("Extracting LUT")
    lut, meta = get_lut(path, processing=not raw)

    print("Extracting isoelastics")
    contours, levels = get_isoelastics(lut, meta, processing=not raw)
    save_iso(path.with_name(path.name.rsplit(".", 1)[0] + "_iso.txt"),
             contours, levels, meta)

    print("Saving LUT")
    if not raw:
        print("...Post-Processing: Removing redundant LUT values.")
        lut = shrink_lut(lut)
    save_lut(path.with_name(path.name.rsplit(
        ".", 1)[0] + "_lut.txt"), lut, meta)
