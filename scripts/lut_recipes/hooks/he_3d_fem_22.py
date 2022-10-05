"""Hooks for the HE-3D-FEM-22 dataset

https://doi.org/10.6084/m9.figshare.20630940.v1

See the individual functions for details.
"""
import matplotlib.pylab as plt
import numpy as np
from scipy import ndimage
from skimage import morphology

from .common import find_upper_intersection


def isoelastics_postprocess(lup, isoel):
    """Process extracted isoelastics"""
    return isoel


def lut_preprocess(lup):
    """Extrapolate isoelasticity lines to convex hull

    The extrapolation has similar results as shown in
    https://github.com/ZELLMECHANIK-DRESDEN/dclab/issues/191.
    """
    xmax = lup.lut_raw[:, 0].max()
    num_isoel = 33

    # Perform linear interpolation until convex hull.
    # don't use small deformation values for isoelastics computation

    lut_iso = lup.lut_raw[
        np.logical_or(lup.lut_raw[:, 1] > 0.03,
                      lup.lut_raw[:, 0] > 120)
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
                inside[ii:ii+4] = True
                break
        parts = new_cc[inside]
        lut_new = np.concatenate((lut_new, parts))
        # Append boundary point
        sx, sy = find_upper_intersection(
            ip, lup.convex_hull, xrange=xrange)
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
