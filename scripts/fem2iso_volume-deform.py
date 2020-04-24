"""Extract volume-deformation isoelasticity lines from simulation data

The simulation dataset must be an HDF5 file with a specific structure
that contains the results of FEM simulations. The original HDF5 files
were provided by Lucas Wittwer.

Creating volume-deformation isoelasticity lines means that a full LUT
is generated from which the isoelasticity lines are then interpolated.
Since the LUT has no function in dclab, it is deleted after the
interpolation step.

The following data post-processing is performed for the LUT (depending
on the HDF5 root attributes):
- If the LUT "model" is "linear elastic" and the LUT "dimensionality"
  is "2Daxis", then the LUT is complemented with analytical values.
  The values for volume are hard-coded and taken from the original
  Matlab scripts. The values for deformation and emodulus are taken
  from "LUT_analytical_linear-elastic_2Daxis.txt". As for the area-
  deformation data, this only affects points with small deformation
  and area below 200um.
- If the LUT "dimensionality" is "2Daxis" (rotationally symmetric
  simulation), then the LUT is cropped at a maximum area of 290um^2.
  The reason is that the axis-symmetric model becomes inaccurate when
  the object boundary comes close to the channel walls (the actual
  flow profile in a rectangular cross-section channel is not anymore
  rotationally symmetric around the object). In addition, there have
  been numerical errors due to meshing if the area is above 290um^2.

The discussion related to this script is archived in issue #70 (dclab).

An example HDF5 file can be found on figshare
(https://doi.org/10.6084/m9.figshare.12155064.v2).
"""
import argparse
import numbers
import pathlib

from dclab.features import emodulus
import h5py
import numpy as np

import fem2lutiso_std


def get_analytical_volume_LUT_2daxis():
    # analytical area_um-defrom LUT
    ap = "LUT_analytical_linear-elastic_2Daxis.txt"
    lut_area = np.loadtxt(pathlib.Path(__file__).parent / ap)
    lut_volume = np.zeros_like(lut_area)
    lut_volume[:, 1] = lut_area[:, 1]
    lut_volume[:, 2] = lut_area[:, 2]

    # FEM data
    here = pathlib.Path(__file__).parent
    anap = here / "LUT_analytical_linear-elastic_2Daxis.txt"
    data, meta = emodulus.load_mtext(anap)
    assert meta["channel_width"] == 20
    assert meta["method"] == "analytical"
    assert meta["dimensionality"] == "2Daxis"

    # BEGIN MATLAB TRANSLATIONS
    # emodulus
    data1 = np.linspace(7.97**(-1/2), 28.43**(-1/2), 23)**(-2)
    raise NotImplementedError("nr_p is not correct, 84?")
    nr_p = 100
    d = 20  # um
    # linear spaced area if assumed a sphere (spaced with square root)
    lambda_rand = np.linspace(0.01, 0.534, nr_p, endpoint=True)**(1/2)
    lambd = lambda_rand[np.abs(lambda_rand-0.5) < 0.5]  # 0<lambd<1
    # END MATLAB TRANSLATIONS
    # In the Matlab script, area in um is computed like this:
    # Area_unitless*1.094^2*d^2*lambda(i)^2/4;
    # (where Area_unitless is from the modeling computations).
    # - The unitless length is the radius of the cylindrical channel
    #   (lambd==1).
    # - The channel width d is always multiplied by the factor 1.094.
    radius = lambd * d/2 * 1.094
    volume = 4/3*np.pi * radius**3

    for emod in data1:
        eloc = np.abs(emod - lut_volume[:, 2]) < .01
        assert np.sum(eloc), "failed to find emodulus {}".format(emod)
        assert np.sum(eloc) == volume.size, "bad size emodulus {}".format(emod)
        lut_volume[eloc, 0] = volume

    return lut_volume


def get_lut_volume(path, processing=True):
    """Extract the volume LUT from an HDF5 file provided by Lucas Wittwer

    Notes
    -----
    - If the LUT "model" is "linear elastic" and the LUT "dimensionality"
      is "2Daxis", then the LUT is complemented with analytical values
      from "LUT_analytical_linear-elastic_2Daxis.txt" (which are extended
      to volume using the volume data used in the original Matlab script
      (see `get_analytical_part_2daxis`)) for small deformation and area
      below 200um. The original FEM simulations did not cover that
      area, because of discretization problems (deformations smaller than
      0.005 could not be resolved).
    - If the LUT "dimensionality" is "2Daxis" (rotationally symmetric
      simulation), then the LUT is cropped at a maximum area of 290um.
      The reason is that the axis-symmetric model becomes inaccurate when
      the object boundary comes close to the channel walls (the actual
      flow profile in a rectangular cross-section channel is not anymore
      rotationally symmetric around the object). In addition, there have
      been numerical errors due to meshing if the area is above 290um^2.
    """
    area_um = []
    volume = []
    deform = []
    emodulus = []

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
                volume.append(sim.attrs["volume"])
                assert sim.attrs["volume_unit"].decode("utf-8") == "um^3"
                deform.append(sim.attrs["deformation"])
                assert sim.attrs["deformation_unit"].decode("utf-8") == ""
                emodulus.append(h5[Ek].attrs["emodulus"]/1000)

    lut = np.zeros((len(emodulus), 4), dtype=float)
    lut[:len(emodulus), 0] = volume
    lut[:len(emodulus), 1] = deform
    lut[:len(emodulus), 2] = emodulus
    lut[:len(emodulus), 3] = area_um

    if processing:
        if meta["dimensionality"] == "2Daxis":
            print("...Post-Processing: Cropping LUT at 290um^2.")
            # the analytical part (below) is anyhow below 200um^2
            lut = lut[lut[:, 3] < 290]

        if (meta["model"] == "linear elastic"
                and meta["dimensionality"] == "2Daxis"):
            print("...Post-Processing: Complementing analytical volume data.")
            # load analytical data
            lut_ana = get_analytical_volume_LUT_2daxis()
            lut = np.concatenate((lut, lut_ana))

    return lut, meta


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
