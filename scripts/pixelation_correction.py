"""Pixelation correction for emodulus look-up-tables and isoelastics

Use this script to study and visualize how pixelation affects
individual features and especially how it affects look-up-tables
for emodulus computation and isoelastics.

This script partially reproduces figures 2, 4, and 5 of Christoph
Herold's famous arXiv manuscript (https://arxiv.org/abs/1704.00572).

As an argument, you should pass a FEM simulation results HDF5 file
(https://doi.org/10.6084/m9.figshare.12155064.v4), which is required
for obtaining a volume-deformation look-up table and for visualizing
pixelation effects using in-silico data (run fem2rtdc.py first).

Please also see dclab issue #70 for a broader view on things.

Notes
-----

I added two example output .png files to this directory:

 - pixelation_correction_2020.png: This is the original output file from
   when this script was first implemented.
 - pixelation_correction_2022.png: This is a revised version from 2022
   which properly implements random offsets in the disk's mask images
   (see dclab issue #178).
"""
import argparse
import os
import pathlib

import dclab
import lmfit
import matplotlib.pylab as plt
import numpy as np


from fem2rtdc import features_from_mask
from lut_recipes import LutProcessor

#: You can edit the pixel size to see how the pixelation effect scales
#: with the pixel size. The fits incorporate this parameter to always
#: yield the same results, but the reference implementations also
#: do this. Not that this does not affect the black dots, which are
#: only plotted for comparison and were generated with fem2rtdc.py.
PIXEL_SIZE = 0.34  # um/px


def get_lut_volume(path):
    lup = LutProcessor(path, use_hooks=True, featx="volume")
    lut, _, _ = lup.assemble_lut_and_isoelastics()
    return lut, lup.meta


def create_multiexp_model(n=5, x=None, x0=None, y=None, decaycoeffs=None,
                          offset=None):
    """Create a multiexponential model with smart initial values for lmfit"""
    modelexpr = ["off"]
    for ii in range(n):
        modelexpr.append("A{} * exp(-(x-x0)/t{})".format(ii, ii))

    mod = lmfit.models.ExpressionModel(" + ".join(modelexpr))
    params = mod.make_params()

    if x is not None:
        params["x0"].set(value=x.min(), min=-np.abs(x).max()/20, max=np.min(x))

    params["off"].set(min=0, max=2, value=.1)

    if x0 is not None:
        params["x0"].set(value=x0, vary=False)

    amplitudes = [.01] * n

    if decaycoeffs is None:
        vary_tau = True
        if x is not None:
            decaycoeffs = np.logspace(-n, 0, n, base=3)/5*(x.max()-x.min())
        else:
            decaycoeffs = np.logspace(0, 4, n)

    else:
        assert len(decaycoeffs) == n
        vary_tau = False

    for ii in range(n):
        if x is not None:
            tmax = x.max() - x.min()
        else:
            tmax = 10**n
        params["t{}".format(ii)].set(min=1e-5, max=tmax,
                                     value=decaycoeffs[ii], vary=vary_tau)
        if y is not None:
            amax = 2*(y.max() - y.min())
        else:
            amax = 1

        params["A{}".format(ii)].set(min=1e-5, max=amax, value=amplitudes[ii])

    if offset is None and y is not None:
        idoff = x > x.max()*.9
        params["off"].set(value=np.mean(y[idoff]))
    else:
        params["off"].set(value=offset, vary=False)

    return mod, params


def disk_image(area_um, random_displacement=True, seed=None):
    """Compute pixelated disk image for a given event area

    The disk is mapped on a grid. A pixel is assumed to be
    inside the disk when its center point is within the
    perimeter of the disk.

    Parameters
    ----------
    area_um: float
        Accurate area [µm²] of the event
    random_displacement: bool
        If set to True, randomly displace the center of the
        disk by up to 1 px in x and y. If set to False, the center of
        the disk is at the center between two pixels.
    seed: int or None
        Random seed to use when moving the disk in the x-y-plane
        before performing the mapping onto the grid. Set to None if
        you have to set the seed someplace else.

    Returns
    -------
    mask: 2D boolean ndarray
        Binary disk image
    """
    # radius in pixels
    radius = np.sqrt(area_um / np.pi) / PIXEL_SIZE

    size = int(np.ceil(radius*2.2))
    if size % 2:
        size += 1
    center = size / 2

    # determine suitable mesh
    x = np.arange(size)
    xm, ym = np.meshgrid(x, x, indexing="ij")

    if not random_displacement:
        # centered between pixels
        offx = 0.5
        offy = 0.5
    else:
        # random offsets
        if seed is not None:
            np.random.seed(seed)
        offx = np.random.uniform(0, 1)
        offy = np.random.uniform(0, 1)
    xm = xm - center + offx
    ym = ym - center + offy

    # binary disk image
    mask = (xm**2 + ym**2) <= radius**2

    return mask


def fit_exponential(dataf, datax, decaycoeffs, x0=None, offset=None):
    """Fit a multi-exponential decay

    This is used for reproducing the pixelation correction.

    Parameters
    ----------
    dataf: 1D ndarray of length N
        The y-data to fit
    datax: 1D ndarray of length N
        The corresponding x data
    decaycoeffs: list of float
        The decay times (coefficients). They are defined
        in `create_multiexp_model`. They are not varied
        during fitting.
    x0: float or None
        If None, then the constant x-offset is a fit parameter.
        If a float, then the constant x-offset is fixed.
    offset: float or None
        If None, then the constant y-offset is a fit parameter
        (initial value determined from `dataf` for large `datax`).
        If a float, then the constant y-offset is fixed.

    Returns
    -------
    result: lmfit fitting results object
    """
    mod, params = create_multiexp_model(n=len(decaycoeffs),
                                        x=datax,
                                        x0=x0,
                                        offset=offset,
                                        decaycoeffs=decaycoeffs,
                                        y=dataf)

    result = mod.fit(dataf, params=params, x=datax)
    return result


def fit_exponential_free(dataf, datax, n, x0=None, offset=None):
    """Fit a multi-exponential decay (prototyping)

    This is used for finding the decay times (coefficients)
    for `fit_exponential`. The initial values of the decay
    times are spaced logarithmically. You can increase `n` and
    then just remove those coefficients that did not vary during
    fitting.

    Parameters
    ----------
    dataf: 1D ndarray of length N
        The y-data to fit
    datax: 1D ndarray of length N
        The corresponding x data
    n: int
        Number of exponentials to fit.
    x0: float or None
        If None, then the constant x-offset is a fit parameter.
        If a float, then the constant x-offset is fixed.
    offset: float or None
        If None, then the constant y-offset is a fit parameter
        (initial value determined from `dataf` for large `datax`).
        If a float, then the constant y-offset is fixed.

    Returns
    -------
    result: lmfit fitting results object
    """
    mod, params = create_multiexp_model(n=n,
                                        x=datax,
                                        offset=offset,
                                        x0=x0,
                                        y=dataf)

    result = mod.fit(dataf, params=params, x=datax)
    return result


def corr_deform_with_area_um(area_um, px_um=PIXEL_SIZE):
    """original from Christoph Herold"""
    pxscale = (.34 / px_um)**2
    offs = 0.0012
    exp1 = 0.020 * np.exp(-area_um * pxscale / 7.1)
    exp2 = 0.010 * np.exp(-area_um * pxscale / 38.6)
    exp3 = 0.005 * np.exp(-area_um * pxscale / 296)
    delta = offs + exp1 + exp2 + exp3
    return delta


def corr_area_um_with_area_um(area_um, px_um=PIXEL_SIZE):
    """from here"""
    pxscale = (.34 / px_um)**2
    offs = 1.006
    exp1 = 0.625 * np.exp(-area_um * pxscale / 3.5)
    exp2 = 0.106 * np.exp(-area_um * pxscale / 20)
    exp3 = 0.041 * np.exp(-area_um * pxscale / 86)
    exp4 = 0.021 * np.exp(-area_um * pxscale / 463)
    delta = offs + exp1 + exp2 + exp3 + exp4
    return delta


def corr_deform_with_volume(volume, px_um=PIXEL_SIZE):
    """from here"""
    pxscalev = (.34 / px_um)**3
    offs = 0.0013
    exp1 = 0.0172 * np.exp(-volume * pxscalev / 40)
    exp2 = 0.0070 * np.exp(-volume * pxscalev / 450)
    exp3 = 0.0032 * np.exp(-volume * pxscalev / 6040)
    delta = offs + exp1 + exp2 + exp3
    return delta


def corr_volume_with_volume(volume, px_um=PIXEL_SIZE):
    """from here"""
    pxscalev = (.34 / px_um)**3
    offs = 1.023
    exp1 = 0.719 * np.exp(-volume * pxscalev / 8.3)
    exp2 = 0.123 * np.exp(-volume * pxscalev / 102)
    exp3 = 0.061 * np.exp(-volume * pxscalev / 885)
    exp4 = 0.035 * np.exp(-volume * pxscalev / 8830)
    delta = offs + exp1 + exp2 + exp3 + exp4
    return delta


def plot_numerical_data(path, ax1, ax2, ax3, ax4):
    """Load and plot numerical data exported using femt2rtdc.py

    Parameters
    ----------
    path: pathlib.Path
        Path to a simulation file (this file is actually
        not used, only the output of fem2rtdc.py).
    ax1: matplotlib.Axis
        deform error versus area_um plot
    ax2: matplotlib.Axis
        area_um error versus area_um plot
    ax3: matplotlib.Axis
        deform error versus volume plot
    ax4: matplotlib.Axis
        volume error versus volume plot
    """
    h5mapped = sorted(path.parent.glob("*.map*.rtdc"))
    h5orig = path.with_suffix(".orig.rtdc")

    dso = dclab.new_dataset(h5orig)
    dsm = [dclab.new_dataset(pp) for pp in h5mapped]

    area_um = np.concatenate([ds["area_um"] for ds in dsm])
    deform = np.concatenate([ds["deform"] for ds in dsm])
    volume = np.concatenate([ds["volume"] for ds in dsm])

    true_area_um = np.concatenate([dso["area_um"]]*5)
    true_deform = np.concatenate([dso["deform"]]*5)
    true_volume = np.concatenate([dso["volume"]]*5)

    delta_deform = deform - true_deform
    delta_area_um = true_area_um / area_um
    delta_volume = true_volume / volume

    plotkw = dict(ms=1, color="k", alpha=.05, label="FEM simulation")

    ax1.plot(area_um, delta_deform, ".", **plotkw)
    ax2.plot(area_um, delta_area_um, ".", **plotkw)
    ax3.plot(volume, delta_deform, ".", **plotkw)
    ax4.plot(volume, delta_volume, ".", **plotkw)


def plot_and_fit_disk_data(ax1, ax2, ax3, ax4):
    """Visualize the pixelation effect for deform, area, and volume

    Reproduces the mostly figures 2 and 4 of Christoph Herold's
    famous arXiv paper and goes beyond that.

    We are only investigating discs/spheres here, because Herold
    already showed that using an ellipse does not really change
    things. In addition, we can overlay the disk plot with in-silico
    data from the FEM dataset (see `plot_numerical_data`).

    Parameters
    ----------
    ax1: matplotlib.Axis
        deform error versus area_um plot
    ax2: matplotlib.Axis
        area_um error versus area_um plot
    ax3: matplotlib.Axis
        deform error versus volume plot
    ax4: matplotlib.Axis
        volume error versus volume plot

    Notes
    -----
    - Computed data are cached as .npz in the script directory.
    - The fitting region goes much beyond the region shown in
      the plot produced by this script (__main__ section).
    - The reference fit for deform-vs-area_um plot is taken
      from Herold's manuscript (this is why it does not fully
      coincide with the fit here).
    """
    # sanity check (circle center positioned at pixel corner)
    assert disk_image(np.pi*(PIXEL_SIZE*2)**2, False).sum() == 12

    N = 10000
    R = 5

    here = pathlib.Path(__file__).parent
    path = here / ".cache_pixelation_correction_{}.npz".format(PIXEL_SIZE)
    if not os.path.exists(path):
        area_um = np.zeros(N*R, dtype=float)
        deform = np.zeros(N*R, dtype=float)
        volume = np.zeros(N*R, dtype=float)

        true_area_um = np.zeros(N*R, dtype=float)
        true_deform = np.zeros(N*R, dtype=float)
        true_volume = np.zeros(N*R, dtype=float)

        for rr in np.arange(R):
            np.random.seed(R)
            for ii, ari in enumerate(np.linspace(10, 1250, N)):
                # Shape-In values
                mask = disk_image(ari)
                feats = features_from_mask(mask, pixel_size=PIXEL_SIZE)
                area_um[rr + ii*R] = feats["area_um"]
                deform[rr + ii*R] = feats["deform"]
                volume[rr + ii*R] = feats["volume"]
                # True values
                true_area_um[rr + ii*R] = ari
                radius = np.sqrt(ari/np.pi)
                true_volume[rr + ii*R] = 4/3 * np.pi * radius**3
                true_deform[rr + ii*R] = 0  # circle

        delta_deform = deform - true_deform
        delta_area_um = true_area_um / area_um
        delta_volume = true_volume / volume

        np.savez(path,
                 delta_deform=delta_deform,
                 delta_area_um=delta_area_um,
                 delta_volume=delta_volume,
                 area_um=area_um,
                 deform=deform,
                 volume=volume,
                 )
    else:
        dd = np.load(path)
        delta_deform = dd["delta_deform"]
        delta_area_um = dd["delta_area_um"]
        delta_volume = dd["delta_volume"]
        area_um = dd["area_um"]
        deform = dd["deform"]
        volume = dd["volume"]

    # Pixelation effects scale with pixel size
    pxscale = (.34 / PIXEL_SIZE)**2
    pxscalev = (.34 / PIXEL_SIZE)**3

    # Figure 2 Herold: deform-area_um
    ax1.set_title("Deviation of deformation (area_um)")
    ax1.set_ylabel("deform - true_deform")

    aref = np.linspace(area_um.min(), area_um.max(), 1000)
    # Fit with fixed a0, b0, c0
    result_deform = fit_exponential(dataf=delta_deform,
                                    datax=area_um * pxscale,
                                    x0=0,
                                    offset=0.0012,
                                    decaycoeffs=[7.1, 38.6, 296])

    # Plot comparison
    ax1.plot(area_um, delta_deform, ".", ms=1, label="sphere")
    ax1.plot(area_um, result_deform.best_fit, label="fit to sphere data")
    ax1.plot(aref, corr_deform_with_area_um(aref), linestyle=(
        0, (5, 10)), color="k", alpha=.5, label="reference fit")

    # Figure 4 (only circle) Herold: area_um-area_um
    ax2.set_title("Deviation of area (area_um)")
    ax2.set_ylabel("true_area_um / area_um")

    result_area_um = fit_exponential(dataf=delta_area_um,
                                     datax=area_um * pxscale,
                                     x0=0,
                                     offset=1.006,
                                     decaycoeffs=[3.5, 20, 86, 463])

    ax2.plot(area_um, delta_area_um, ".", ms=1, label="sphere")
    ax2.plot(area_um, result_area_um.best_fit, label="fit to sphere data")
    ax2.plot(aref, corr_area_um_with_area_um(aref), linestyle=(
        0, (5, 10)), color="k", alpha=.5, label="reference fit")

    # New figure deform-volume
    ax3.set_title("Deviation of deformation (volume)")
    ax3.set_ylabel("deform - true_deform")

    vref = np.linspace(volume.min(), volume.max(), 1000)
    result_deform_v = fit_exponential(dataf=delta_deform,
                                      datax=volume * pxscalev,
                                      x0=0,
                                      offset=0.0013,
                                      decaycoeffs=[40, 450, 6040])

    ax3.plot(volume, delta_deform, ".", ms=1, label="sphere")
    ax3.plot(volume, result_deform_v.best_fit, label="fit to sphere data")
    ax3.plot(vref, corr_deform_with_volume(vref), linestyle=(
        0, (5, 10)), color="k", alpha=.5, label="reference fit")

    # New figure volume-volume
    ax4.set_title("Deviation of volume (volume)")
    ax4.set_ylabel("true_volume / volume")

    result_volume = fit_exponential(dataf=delta_volume,
                                    datax=volume * pxscalev,
                                    x0=0,
                                    offset=1.023,
                                    decaycoeffs=[8.3, 102, 885, 8830])

    ax4.plot(volume, delta_volume, ".", ms=1, label="sphere")
    ax4.plot(volume, result_volume.best_fit, label="fit to sphere data")
    ax4.plot(vref, corr_volume_with_volume(vref), linestyle=(
        0, (5, 10)), color="k", alpha=.5, label="reference fit")


def plot_emodulus_deviation_dependency(path, ax5, ax6, ax7, ax8):
    """Visualize the effect of pixelation on emodulus

    Reproduces figure 5 of Herold's arXiv paper and goes
    beyond that.

    The idea is to add the pixelation correction (which would
    otherwise be subtracted) before interpolatimg the emodulus
    and then visualize by how far we are off compared to when
    we don't add the pixelation correction.

    Parameters
    ----------
    path: pathlib.Path
        Path to a FEM simulation results file. This is
        used for generating the deform-volume LUT
    ax5: matplotlib.Axis
        overestimation of emodulus for deform-vs-area_um
    ax6: matplotlib.Axis
        overestimation of emodulus for area_um-vs-area_um
    ax7: matplotlib.Axis
        overestimation of emodulus for deform-vs-volume
    ax8: matplotlib.Axis
        overestimation of emodulus for volume-vs-volume

    Notes
    -----
    - The relative error for area_um and volume are negative.
    - The low relative errors for area_um and volume tell us
      that we don't need a pixelation correction here. This
      also becomes obvious when looking at the LUT: A change
      in area (move horizontally through the LUT) does not
      change the emodulus much.
    - The default dclab LUT is used for deform-area_um data
    """
    # Plot expected deviations
    # deform-area_um
    ax5.set_ylabel("relative error (1-E/E0)")
    area_um = np.linspace(5, 290, 200)
    emodkw = dict(
        area_um=area_um,
        medium=15,
        channel_width=20,
        flow_rate=0.04,
        px_um=0,
        temperature=None)
    for dv in [0.005, 0.01, 0.02, 0.03, 0.1]:
        deform = np.ones_like(area_um) * dv
        emod0 = dclab.features.emodulus.get_emodulus(
            deform=deform, **emodkw)
        pxcorr = corr_deform_with_area_um(area_um)
        emod1 = dclab.features.emodulus.get_emodulus(
            deform=deform+pxcorr, **emodkw)
        ax5.plot(area_um, 1-emod1/emod0, label="{}".format(dv))

    # area_um-area_um
    ax6.set_ylabel("negative relative error -(1-E/E0)")
    emodkw = dict(
        medium=15,
        channel_width=20,
        flow_rate=0.04,
        px_um=0,
        temperature=None)
    for dv in [0.005, 0.01, 0.02, 0.03, 0.1]:
        deform = np.ones_like(area_um) * dv
        emod0 = dclab.features.emodulus.get_emodulus(
            deform=deform, area_um=area_um, **emodkw)
        pxcorr = corr_area_um_with_area_um(area_um)
        emod1 = dclab.features.emodulus.get_emodulus(
            deform=deform, area_um=area_um+pxcorr, **emodkw)
        ax6.plot(area_um, -(1-emod1/emod0), label="{}".format(dv))

    # deform-volume
    ax7.set_ylabel("relative error (1-E/E0)")
    lut_data_volume = get_lut_volume(path=path)
    volume = np.linspace(50, 3200, 200)
    emodkwv = dict(
        volume=volume,
        medium=15,
        channel_width=20,
        flow_rate=0.04,
        px_um=0,
        temperature=None,
        lut_data=lut_data_volume)
    for dv in [0.005, 0.01, 0.02, 0.03, 0.1]:
        deform = np.ones_like(volume) * dv
        emod0 = dclab.features.emodulus.get_emodulus(
            deform=deform, **emodkwv)
        pxcorr = corr_deform_with_volume(volume)
        emod1 = dclab.features.emodulus.get_emodulus(
            deform=deform+pxcorr, **emodkwv)
        ax7.plot(volume, 1-emod1/emod0, label="{}".format(dv))

    # volume-volume
    ax8.set_ylabel("negative relative error -(1-E/E0)")
    lut_data_volume = get_lut_volume(path=path)
    volume = np.linspace(50, 3200, 200)
    emodkwv = dict(
        medium=15,
        channel_width=20,
        flow_rate=0.04,
        px_um=0,
        temperature=None,
        lut_data=lut_data_volume)
    for dv in [0.005, 0.01, 0.02, 0.03, 0.1]:
        deform = np.ones_like(volume) * dv
        emod0 = dclab.features.emodulus.get_emodulus(
            deform=deform, volume=volume, **emodkwv)
        pxcorr = corr_volume_with_volume(volume)
        emod1 = dclab.features.emodulus.get_emodulus(
            deform=deform, volume=volume+pxcorr, **emodkwv)
        ax8.plot(volume, -(1-emod1/emod0), label="{}".format(dv))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("fempath",
                        help="Path to an .hdf5 FEM simulation file",
                        )
    args = parser.parse_args()
    path = pathlib.Path(args.fempath)

    fig = plt.figure(figsize=(15, 10))
    # plots for visualizing the pixelation effect
    ax1 = plt.subplot(241)
    ax2 = plt.subplot(242)
    ax3 = plt.subplot(243)
    ax4 = plt.subplot(244)
    # plots for visualizing the effect of pixelation on emodulus
    ax5 = plt.subplot(245)
    ax6 = plt.subplot(246)
    ax7 = plt.subplot(247)
    ax8 = plt.subplot(248)

    plot_numerical_data(path, ax1, ax2, ax3, ax4)
    plot_and_fit_disk_data(ax1, ax2, ax3, ax4)
    plot_emodulus_deviation_dependency(path, ax5, ax6, ax7, ax8)

    for ax in [ax1, ax2, ax5, ax6]:
        ax.set_xlim(0, 290)
        ax.set_xlabel("area_um [µm²]")

    for ax in [ax3, ax4, ax7, ax8]:
        ax.set_xlim(0, 3200)
        ax.set_xlabel("volume [µm³]")

    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend()
        ax.grid()

    for ax in [ax5, ax6, ax7, ax8]:
        ax.grid()
        ax.legend(title="deformation")
        ax.set_ylim(1e-4, 1)
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(f"{path.stem}_pixelation_correction_gen_{PIXEL_SIZE}.png",
                dpi=150)
    plt.show()
