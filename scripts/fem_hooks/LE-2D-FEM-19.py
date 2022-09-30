"""Hooks for the LE-2D-FEM-19 dataset

https://doi.org/10.6084/m9.figshare.12155064.v4
See the individual functions for details.
"""
import pathlib

import numpy as np

from dclab.features import emodulus


def get_analytical_volume_lut_2daxis():
    """Compute the volume-deformation analytical part of the LUT

    The data stored in LUT_analytical_linear-elastic_2Daxis.txt
    do not include the volume information (only the cross-sectional
    area of the deformed sphere). Since the linear elastic model
    means that volume is conserved, we can just compute the volume
    data by redoing the computations in the original Matlab script
    (CreateAvsChist_Loop_CH4.m). In addition, we have to crop the
    first 15 and the last data points which were manually removed
    to better fit in with the numerical values.
    """
    # analytical area_um-deform LUT
    ap = "LUT_analytical_linear-elastic_2Daxis.txt"
    lut_area = np.loadtxt(pathlib.Path(__file__).parent / ap)
    lut_volume = np.zeros_like(lut_area)
    lut_volume[:, 1] = lut_area[:, 1]
    lut_volume[:, 2] = lut_area[:, 2]

    # FEM data
    here = pathlib.Path(__file__).parent
    anap = here / "LUT_analytical_linear-elastic_2Daxis.txt"
    _, meta = emodulus.load_mtext(anap)
    assert meta["channel_width"] == 20
    assert meta["method"] == "analytical"
    assert meta["dimensionality"] == "2Daxis"

    # BEGIN MATLAB TRANSLATIONS
    # emodulus
    data1 = np.linspace(7.97**(-1/2), 28.43**(-1/2), 23)**(-2)
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

    # The data stored in LUT_analytical_linear-elastic_2Daxis.txt does not
    # contain the full nr_p=100 points, but it was cropped manually *sigh*.
    # By manual inspection of of the highest emodulus isoelasticity line
    # and comparison with area=np.pi*radius**2, I am quite certain that the
    # first 15 data points and the last datapoint were cropped.
    volume = volume[15:-1]

    for emod in data1:
        eloc = np.abs(emod - lut_volume[:, 2]) < .01
        assert np.sum(eloc), "failed to find emodulus {}".format(emod)
        assert np.sum(eloc) == volume.size, "bad size emodulus {}".format(emod)
        lut_volume[eloc, 0] = volume

    return lut_volume


def process_isoelastics(contour_lines):
    """Nothing to do since 0.46.0"""
    return contour_lines


def process_lut_areaum_deform(lut):
    """Complement FEM LUT with analytical values and crop it

    The LUT is complemented with analytical values from
    "LUT_analytical_linear-elastic_2Daxis.txt" for small deformation
    and area below 200um. The original FEM simulations did not cover that
    area, because of discretization problems (deformations smaller than
    0.005 could not be resolved).

    The LUT is cropped at a maximum area of 290um^2. The reason is that
    the axis-symmetric model becomes inaccurate when the object boundary
    comes close to the channel walls (the actual flow profile in a
    rectangular cross-section channel is not anymore rotationally symmetric
    around the object). In addition, there have been numerical errors due
    to meshing if the area is above 290um^2.
    """
    ap = "LUT_analytical_linear-elastic_2Daxis.txt"
    print(f"...Post-Processing: Adding analytical LUT from {ap}.")
    # load analytical data
    lut_ana = np.loadtxt(pathlib.Path(__file__).parent / ap)
    lut = np.concatenate((lut, lut_ana))

    print("...Post-Processing: Cropping LUT at 290um^2.")
    lut = lut[lut[:, 0] < 290]
    return lut


def process_lut_volume_deform(lut):
    """Complement FEM LUT with analytical values and crop it

    The LUT is complemented with analytical values from
    "LUT_analytical_linear-elastic_2Daxis.txt" (which are extended
    to volume using the volume data used in the original Matlab script
    (see `get_analytical_part_2daxis`)) for small deformation and area
    below 200um. The original FEM simulations did not cover that
    area, because of discretization problems (deformations smaller than
    0.005 could not be resolved).

    The LUT is cropped at a maximum volume of 3200um^3. The reason is
    that the axis-symmetric model becomes inaccurate when the object
    boundary comes close to the channel walls (the actual flow profile
    in a rectangular cross-section channel is not anymore rotationally
    symmetric around the object). In addition, there have been numerical
    errors due to meshing if the area is above 290um^2.
    """
    print("...Post-Processing: Cropping LUT at 3200um^3.")
    # the analytical part (below) is anyhow below 200um^2
    # We cannot crop at 290um^2, because this will result in
    # a convex lut with interpolation taking place within it.
    # Converting the 290 to an equivalent sphere volume results
    # in a value outside the lut (3700 something). So we just
    # guess a value here:
    lut = lut[lut[:, 0] < 3200, :]

    print("...Post-Processing: Complementing analytical volume data.")
    # load analytical data
    lut_ana = get_analytical_volume_lut_2daxis()
    lut = np.concatenate((lut, lut_ana))

    return lut
