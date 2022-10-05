"""Hooks for the LE-2D-FEM-19 dataset

https://doi.org/10.6084/m9.figshare.12155064.v4

See the individual functions for details.
"""
import pathlib

import matplotlib.pylab as plt
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage import morphology

from dclab.features import emodulus

from .common import find_upper_intersection


def isoelastics_postprocess(lup, isoel):
    """Process extracted isoelastics"""
    new_isoel = []
    for ie in isoel:
        # remove noise at beginning
        if lup.featx == "area_um":
            xmin = 34.93
        elif lup.featx == "volume":
            xmin = 160
        else:
            raise NotImplementedError(f"Unexpected axis {lup.featx}!")
        ie = ie[ie[:, 0] > xmin]
        ie = ie[ie[:, 1] < .187]
        new_isoel.append(ie)
    return new_isoel


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


def lut_preprocess(lup):
    """Complement FEM LUT with analytical values, extrapolate, and crop it

    The LUT is complemented with analytical values from
    "LUT_analytical_linear-elastic_2Daxis.txt" for small deformation
    and area below 200um. The original FEM simulations did not cover that
    area, because of discretization problems (deformations smaller than
    0.005 could not be resolved).

    A sparse region for larger deformation is complemented with
    extrapolated isoelasticity lines (3rd order polynomial fit).
    See https://github.com/ZELLMECHANIK-DRESDEN/dclab/issues/191.

    The LUT is cropped at a maximum area of 290um^2. The reason is that
    the axis-symmetric model becomes inaccurate when the object boundary
    comes close to the channel walls (the actual flow profile in a
    rectangular cross-section channel is not anymore rotationally symmetric
    around the object). In addition, there have been numerical errors due
    to meshing if the area is above 290um^2.
    """
    if lup.featx == "area_um" and lup.featy == "deform":
        # Add additional values from analytical simulations and crop.
        lup.lut_raw = lut_preprocess_area_um_deform(lup)
        xcrop = 170
        xmax = 300
        num_isoel = 30
    elif lup.featx == "volume" and lup.featy == "deform":
        lup.lut_raw = lut_preprocess_volume_deform(lup)
        xcrop = 2300
        xmax = 3450
        num_isoel = 35
    else:
        raise NotImplementedError("Cannot process")

    # Perform linear interpolation until convex hull.
    # don't use small deformation values for isoelastics computation
    lut_iso = lup.lut_raw[
        np.logical_or(lup.lut_raw[:, 1] > 0.05,
                      lup.lut_raw[:, 0] > xcrop)
    ]

    emod, mask_sim = lup.map_lut_to_grid(lut_iso)
    # Find points that should not be in that 2D `emod` array.
    # (bad interpolation from convex vs raw enclosing polygon)
    # Apply a closing disk filter
    # Zero-pad the mask beforehand (otherwise the disk filter is not
    # properly applied to pixels at the boundary of the image)
    ds = 20  # disk closing size
    mask_padded = np.pad(mask_sim, ((ds, ds), (ds, ds)))
    mask_padded_disk = morphology.binary_closing(
        mask_padded, footprint=morphology.disk(ds))
    # Fill any holes (in case of sparse simulations)
    ndimage.binary_fill_holes(mask_padded_disk, output=mask_padded_disk)
    mask_disk = mask_padded_disk[ds:-ds, ds:-ds]

    emod[~mask_disk] = np.nan
    levels, contours, contours_px = lup.extract_isoelastics_from_grid(
        emod, lut=lut_iso, num=num_isoel)

    # Extend the LUT via extrapolation
    lut_new = lup.lut_raw
    ncs = []
    if lup.verbose:
        ax1 = plt.subplot(121, title="extrapolation region")
        ax1.imshow(1.*mask_disk + mask_sim)
        ax2 = plt.subplot(122, title="LUT extrapolation")
        plt.plot(lup.lut_raw[:, 0], lup.lut_raw[:, 1], ".")
    for lev, cc in zip(levels, contours):
        if np.isnan(lev) or len(cc) < 10:
            continue
        ip = np.polynomial.Polynomial.fit(cc[:, 0], cc[:, 1], 3)
        new_cc = np.zeros((20, 3))
        xrange = (cc[-1, 0], 1.2*cc[-1, 0])
        new_cc[:, 0] = np.linspace(xrange[0], xrange[1], 20)
        new_cc[:, 1] = ip(new_cc[:, 0])
        new_cc[:, 2] = lev
        inside = lup.points_in_convex_hull(new_cc[:, :2])
        # Add a few points outside the convex hull for isoelastics computation
        for ii in range(len(inside)):
            if not inside[ii]:
                inside[ii:ii+3] = True
                break
        parts = new_cc[inside]
        lut_new = np.concatenate((lut_new, parts))
        # Append boundary point
        try:
            sx, sy = find_upper_intersection(
                ip, lup.convex_hull, xrange=xrange)
        except BaseException:
            continue
        else:
            new_cc = np.concatenate((new_cc, [[sx, sy, lev]]))
            ncs.append(new_cc)

        if lup.verbose:
            xtest = np.linspace(cc[0, 0], 1.5*cc[-1, 0], 20, endpoint=True)
            ax2.plot(cc[:, 0], cc[:, 1], "o")
            ax2.plot(xtest, ip(xtest))
            ax2.plot(new_cc[:, 0], new_cc[:, 1], "x")

    if lup.verbose:
        ax2.plot(lup.convex_hull[:, 0], lup.convex_hull[:, 1])
        ax2.set_ylim(0, 0.2)
        ax2.set_xlim(0, xmax)
        plt.show()
    lup.lut_raw = lut_new


def lut_preprocess_area_um_deform(lup):
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
    lut = lup.lut_raw
    ap = "LUT_analytical_linear-elastic_2Daxis.txt"
    print(f"...Post-Processing: Adding analytical LUT from {ap}.")
    # load analytical data
    lut_ana = np.loadtxt(pathlib.Path(__file__).parent / ap)
    lut = np.concatenate((lut, lut_ana))

    print("...Post-Processing: Cropping LUT at 290um^2.")
    lut = lut[lut[:, 0] < 290]
    lup.lut_raw = lut

    # Compute new convex hull of LUT
    hull = ConvexHull(lut[:, :2])
    lup.convex_hull = hull.points[hull.vertices, :]

    return lut


def lut_preprocess_volume_deform(lup):
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
    print("...Post-Processing: Complementing analytical volume data.")
    # load analytical data
    lut_ana = get_analytical_volume_lut_2daxis()
    lut = np.concatenate((lup.lut_raw, lut_ana))

    print("...Post-Processing: Cropping LUT at 3200um^3.")
    # the analytical part (below) is anyhow below 200um^2
    # We cannot crop at 290um^2, because this will result in
    # a convex lut with interpolation taking place within it.
    # Converting the 290 to an equivalent sphere volume results
    # in a value outside the lut (3700 something). So we just
    # guess a value here:
    lut = lut[lut[:, 0] < 3200, :]
    lup.lut_raw = lut

    hull = ConvexHull(lut[:, :2])
    lup.convex_hull = hull.points[hull.vertices, :]

    return lut
