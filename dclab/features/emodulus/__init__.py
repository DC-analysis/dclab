#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Computation of apparent Young's modulus for RT-DC measurements"""
from __future__ import division, print_function, unicode_literals

import json
import numbers
import pathlib
from pkg_resources import resource_filename
import warnings

import numpy as np
import scipy.interpolate as spint

from ... import definitions as dfn
from .viscosity import get_viscosity


#: Set this to True to globally enable spline extrapolation when the
#: `area_um`/`deform` data are outside of a LUT. This is discouraged and
#: a :class:`KnowWhatYouAreDoingWarning` warning will be issued.
INACCURATE_SPLINE_EXTRAPOLATION = False

#: Dictionary of look-up tables shipped with dclab.
INTERNAL_LUTS = {
    "FEM-2Daxis": "emodulus_lut.txt",
    }


class KnowWhatYouAreDoingWarning(UserWarning):
    pass


def convert(area_um, deform, channel_width_in, channel_width_out,
            emodulus=None, flow_rate_in=None, flow_rate_out=None,
            viscosity_in=None, viscosity_out=None, inplace=False):
    """convert area-deformation-emodulus triplet

    The conversion formula is described in :cite:`Mietke2015`.

    Parameters
    ----------
    area_um: ndarray
        Convex cell area [µm²]
    deform: ndarray
        Deformation
    channel_width_in: float
        Original channel width [µm]
    channel_width_out: float
        Target channel width [µm]
    emodulus: ndarray
        Young's Modulus [kPa]
    flow_rate_in: float
        Original flow rate [µL/s]
    flow_rate_out: float
        Target flow rate [µL/s]
    viscosity_in: float
        Original viscosity [mPa*s]
    viscosity_out: float or ndarray
        Target viscosity [mPa*s]; This can be an array
    inplace: bool
        If True, override input arrays with corrected data

    Returns
    -------
    area_um_corr: ndarray
        Corrected cell area [µm²]
    deform_corr: ndarray
        Deformation (a copy if `inplace` is False)
    emodulus_corr: ndarray
        Corrected emodulus [kPa]; only returned if `emodulus` is given.

    Notes
    -----
    If only `area_um`, `deform`, `channel_width_in` and
    `channel_width_out` are given, then only the area is
    corrected and returned together with the original deform.
    If all other arguments are not set to None, the emodulus
    is corrected and returned as well.
    """
    warnings.warn("Usage of the `convert` method is deprecated! Please use "
                  + "the scale_feature method instead.")
    copy = not inplace

    deform_corr = np.array(deform, copy=copy)

    if channel_width_in != channel_width_out:
        area_um_corr = scale_area_um(area_um, channel_width_in,
                                     channel_width_out, inplace)
    else:
        area_um_corr = np.array(area_um, copy=copy)

    if (emodulus is not None
        and flow_rate_in is not None
        and flow_rate_out is not None
        and viscosity_in is not None
            and viscosity_out is not None):
        emodulus_corr = scale_emodulus(emodulus, channel_width_in,
                                       channel_width_out, flow_rate_in,
                                       flow_rate_out, viscosity_in,
                                       viscosity_out, inplace)

    if emodulus is None:
        return area_um_corr, deform_corr
    else:
        return area_um_corr, deform_corr, emodulus_corr


def corrpix_deform_delta(area_um, px_um=0.34):
    """Deformation correction term for pixelation effects

    The contour in RT-DC measurements is computed on a
    pixelated grid. Due to sampling problems, the measured
    deformation is overestimated and must be corrected.

    The correction formula is described in :cite:`Herold2017`.

    Parameters
    ----------
    area_um: float or ndarray
        Apparent (2D image) area in µm² of the event(s)
    px_um: float
        The detector pixel size in µm.
    inplace: bool
        Change the deformation values in-place

    Returns
    -------
    deform_delta: float or ndarray
        Error of the deformation of the event(s) that must be
        subtracted from `deform`.
        deform_corr = deform -  deform_delta
    """
    # A triple-exponential decay can be used to correct for pixelation
    # for apparent cell areas between 10 and 1250µm².
    # For 99 different radii between 0.4 μm and 20 μm circular objects were
    # simulated on a pixel grid with the pixel resolution of 340 nm/pix. At
    # each radius 1000 random starting points were created and the
    # obtained contours were analyzed in the same fashion as RT-DC data.
    # A convex hull on the contour was used to calculate the size (as area)
    # and the deformation.
    # The pixel size correction `pxcorr` takes into account the pixel size
    # in the pixelation correction formula.
    pxcorr = (.34 / px_um)**2
    offs = 0.0012
    exp1 = 0.020 * np.exp(-area_um * pxcorr / 7.1)
    exp2 = 0.010 * np.exp(-area_um * pxcorr / 38.6)
    exp3 = 0.005 * np.exp(-area_um * pxcorr / 296)
    delta = offs + exp1 + exp2 + exp3

    return delta


def extrapolate_emodulus(lut, datax, deform, emod, deform_norm,
                         deform_thresh=.05, inplace=True):
    """Use spline interpolation to fill in nan-values

    When points (`datax`, `deform`) are outside the convex
    hull of the lut, then :func:`scipy.interpolate.griddata` returns
    nan-valules.

    With this function, some of these nan-values are extrapolated
    using :class:`scipy.interpolate.SmoothBivariateSpline`. The
    supported extrapolation values are currently limited to those
    where the deformation is above 0.05.

    A warning will be issued, because this is not really
    recommended.

    Parameters
    ----------
    lut: ndarray of shape (N, 3)
        The normalized (!! see :func:`normalize`) LUT (first axis is
        points, second axis enumerates datax, deform, and emodulus)
    datax: ndarray of size N
        The normalized x data (corresponding to `lut[:, 0]`)
    deform: ndarray of size N
        The normalized deform (corresponding to `lut[:, 1]`)
    emod: ndarray of size N
        The emodulus (corresponding to `lut[:, 2]`); If `emod`
        does not contain nan-values, there is nothing to do here.
    deform_norm: float
        The normalization value used to normalize `lut[:, 1]` and
        `deform`.
    deform_thresh: float
        Not the entire LUT is used for bivariate spline interpolation.
        Only the points where `lut[:, 1] > deform_thresh/deform_norm`
        are used. This is necessary, because for small deformations,
        the LUT has an extreme slope that kills any meaningful
        spline interpolation.
    """
    if not inplace:
        emod = np.array(emod, copy=True)
    # unknowns are nans and deformation values above the threshold
    unkn = np.logical_and(np.isnan(emod),
                          deform > deform_thresh/deform_norm)

    if np.sum(unkn) == 0:
        # nothing to do
        return emod

    warnings.warn("LUT extrapolation is barely tested and may yield "
                  + "unphysical values!",
                  KnowWhatYouAreDoingWarning)

    lut_crop = lut[lut[:, 1] > deform_thresh/deform_norm, :]

    itp = spint.SmoothBivariateSpline(x=lut_crop[:, 0],
                                      y=lut_crop[:, 1],
                                      z=lut_crop[:, 2],
                                      )

    emod[unkn] = itp.ev(datax[unkn], deform[unkn])
    return emod


def get_emodulus(area_um=None, deform=None, volume=None, medium="CellCarrier",
                 channel_width=20.0, flow_rate=0.16, px_um=0.34,
                 temperature=23.0, lut_data="FEM-2Daxis",
                 extrapolate=INACCURATE_SPLINE_EXTRAPOLATION, copy=True):
    """Compute apparent Young's modulus using a look-up table

    Parameters
    ----------
    area_um: float or ndarray
        Apparent (2D image) area [µm²] of the event(s)
    deform: float or ndarray
        Deformation (1-circularity) of the event(s)
    volume: float or ndarray
        Apparent volume of the event(s). It is not possible to define
        `volume` and `area_um` at the same time (makes no sense).

        .. versionadded:: 0.25.0
    medium: str or float
        The medium to compute the viscosity for. If a string
        is given, the viscosity is computed. If a float is given,
        this value is used as the viscosity in mPa*s (Note that
        `temperature` must be set to None in this case).
    channel_width: float
        The channel width [µm]
    flow_rate: float
        Flow rate [µL/s]
    px_um: float
        The detector pixel size [µm] used for pixelation correction.
        Set to zero to disable.
    temperature: float, ndarray, or None
        Temperature [°C] of the event(s)
    lut_data: path, str, or tuple of (np.ndarray of shape (N, 3), dict)
        The LUT data to use. If it is a key in :const:`INTERNAL_LUTS`,
        then the respective LUT will be used. Otherwise, a path to a
        file on disk or a tuple (LUT array, meta data) is possible.
        The LUT meta data is used to check whether the given features
        (e.g. `area_um` and `deform`) are valid interpolation choices.

        .. versionadded:: 0.25.0
    extrapolate: bool
        Perform extrapolation using :func:`extrapolate_emodulus`. This
        is discouraged!
    copy: bool
        Copy input arrays. If set to false, input arrays are
        overridden.

    Returns
    -------
    elasticity: float or ndarray
        Apparent Young's modulus in kPa

    Notes
    -----
    - The look-up table used was computed with finite elements methods
      according to :cite:`Mokbel2017` and complemented with analytical
      isoelastics from :cite:`Mietke2015`. The original simulation
      results are available on figshare :cite:`FigshareWittwer2020`.
    - The computation of the Young's modulus takes into account
      corrections for the viscosity (medium, channel width, flow rate,
      and temperature) :cite:`Mietke2015` and corrections for
      pixelation of the area and the deformation which are computed
      from a (pixelated) image :cite:`Herold2017`.
    - By using external LUTs, it is possible to interpolate on the
      volume-deformation plane. This feature was added in version
      0.25.0.

    See Also
    --------
    dclab.features.emodulus.viscosity.get_viscosity: compute viscosity
        for known media
    """
    # Get lut data
    lut, lut_meta = load_lut(lut_data)
    # copy deform so we can use in-place pixel correction
    deform = np.array(deform, dtype=float, copy=copy)
    # Get the correct features (sanity checks)
    featx, featy, _ = lut_meta["column features"]
    if featx == "area_um" and featy == "deform":
        assert volume is None, "Don't define area_um and volume at same time!"
        datax = np.array(area_um, dtype=float, copy=copy)
        if px_um:
            # Correct deformation for pixelation effect (subtract ddelt).
            ddelt = corrpix_deform_delta(area_um=area_um, px_um=px_um)
            deform -= ddelt
    elif featx == "volume" and featy == "deform":
        assert area_um is None, "Don't define area_um and volume at same time!"
        datax = np.array(volume, dtype=float, copy=copy)
        if px_um:
            raise NotImplementedError(
                "No pixelation correction for volume-deform!")
    else:
        raise KeyError("No recipe for '{}' and '{}'!".format(featx, featy))

    # Compute viscosity
    if isinstance(medium, numbers.Number):
        visco = medium
        if temperature is not None:
            raise ValueError("If `medium` is given in Pa*s, then "
                             + "`temperature` must be set to None!")
    else:
        visco = get_viscosity(medium=medium, channel_width=channel_width,
                              flow_rate=flow_rate, temperature=temperature)

    if isinstance(visco, np.ndarray):
        # New in dclab 0.20.0: Computation for viscosities array
        # Convert the input area_um to that of the LUT (deform does not change)
        scale_kw = {"channel_width_in": channel_width,
                    "channel_width_out": lut_meta["channel_width"],
                    "inplace": False}
        datax_4lut = scale_feature(feat=featx, data=datax, **scale_kw)
        deform_4lut = np.array(deform, dtype=float, copy=copy)

        # Normalize interpolation data such that the spacing for
        # area and deformation is about the same during interpolation.
        featx_norm = lut[:, 0].max()
        normalize(lut[:, 0], featx_norm)
        normalize(datax_4lut, featx_norm)

        defo_norm = lut[:, 1].max()
        normalize(lut[:, 1], defo_norm)
        normalize(deform_4lut, defo_norm)

        # Perform interpolation
        emod = spint.griddata((lut[:, 0], lut[:, 1]), lut[:, 2],
                              (datax_4lut, deform_4lut),
                              method='linear')

        if extrapolate:
            # New in dclab 0.23.0: Perform extrapolation outside of the LUT
            # This is not well-tested and thus discouraged!
            extrapolate_emodulus(lut=lut,
                                 datax=datax_4lut,
                                 deform=deform_4lut,
                                 emod=emod,
                                 deform_norm=defo_norm,
                                 inplace=True)

        # Convert the LUT-interpolated emodulus back
        backscale_kw = {"channel_width_in": lut_meta["channel_width"],
                        "channel_width_out": channel_width,
                        "flow_rate_in": lut_meta["flow_rate"],
                        "flow_rate_out": flow_rate,
                        "viscosity_in": lut_meta["fluid_viscosity"],
                        "viscosity_out": visco,
                        "inplace": True}
        # deformation is not scaled (no units)
        scale_feature(feat=featx, data=datax_4lut, **backscale_kw)
        scale_emodulus(emod, **backscale_kw)
    else:
        # Corrections
        # We correct the LUT, because it contains less points than
        # the event data. Furthermore, the lut could be cached
        # in the future, if this takes up a lot of time.
        scale_kw = {"channel_width_in": lut_meta["channel_width"],
                    "channel_width_out": channel_width,
                    "flow_rate_in": lut_meta["flow_rate"],
                    "flow_rate_out": flow_rate,
                    "viscosity_in": lut_meta["fluid_viscosity"],
                    "viscosity_out": visco,
                    "inplace": True}
        # deformation is not scaled (no units)
        scale_feature(feat=featx, data=lut[:, 0], **scale_kw)
        scale_emodulus(lut[:, 2], **scale_kw)

        # Normalize interpolation data such that the spacing for
        # area and deformation is about the same during interpolation.
        featx_norm = lut[:, 0].max()
        normalize(lut[:, 0], featx_norm)
        normalize(datax, featx_norm)

        defo_norm = lut[:, 1].max()
        normalize(lut[:, 1], defo_norm)
        normalize(deform, defo_norm)

        # Perform interpolation
        emod = spint.griddata((lut[:, 0], lut[:, 1]), lut[:, 2],
                              (datax, deform),
                              method='linear')

        if extrapolate:
            # New in dclab 0.23.0: Perform extrapolation outside of the LUT
            # This is not well-tested and thus discouraged!
            extrapolate_emodulus(lut=lut,
                                 datax=datax,
                                 deform=deform,
                                 emod=emod,
                                 deform_norm=defo_norm,
                                 inplace=True)

    return emod


def load_lut(lut_data="FEM-2Daxis"):
    """Load LUT data from disk

    Parameters
    ----------
    lut_data: path, str, or tuple of (np.ndarray of shape (N, 3), dict)
        The LUT data to use. If it is a key in :const:`INTERNAL_LUTS`,
        then the respective LUT will be used. Otherwise, a path to a
        file on disk or a tuple (LUT array, meta data) is possible.

    Returns
    -------
    lut: np.ndarray of shape (N, 3)
        The LUT data for interpolation
    meta: dict
        The LUT metadata

    Notes
    -----
    If lut_data is a tuple of (lut, meta), then nothing is actually
    done (this is implemented for user convenience).
    """
    if lut_data in INTERNAL_LUTS:
        lut_path = resource_filename("dclab.features.emodulus",
                                     INTERNAL_LUTS[lut_data])
        lut, meta = load_mtext(lut_path)
    elif isinstance(lut_data, tuple):
        lut, meta = lut_data
    elif (isinstance(lut_data, (str, pathlib.Path))
          and pathlib.Path(lut_data).exists()):
        lut, meta = load_mtext(lut_data)
    else:
        raise ValueError("`name_path_arr` must be path, key, or array, "
                         "got '{}'!".format(lut_data))
    return lut, meta


def load_mtext(path):
    """Load colunm-based data from text files with metadata

    This file format is used for isoelasticity lines and look-up
    table data in dclab.

    The text file is loaded with `numpy.loadtxt`. The metadata
    are stored as a json task between the "BEGIN METADATA" and
    the "END METADATA" tags. The last comment (#) line before the
    actual data defines the features with units in square
    brackets and tab-separated. For instance:

        # [...]
        #
        # BEGIN METADATA
        # {
        #   "authors": "A. Mietke, C. Herold, J. Guck",
        #   "channel_width": 20.0,
        #   "channel_width_unit": "um",
        #   "date": "2018-01-30",
        #   "dimensionality": "2Daxis",
        #   "flow_rate": 0.04,
        #   "flow_rate_unit": "uL/s",
        #   "fluid_viscosity": 15.0,
        #   "fluid_viscosity_unit": "mPa s",
        #   "method": "analytical",
        #   "model": "linear elastic",
        #   "publication": "https://doi.org/10.1016/j.bpj.2015.09.006",
        #   "software": "custom Matlab code",
        #   "summary": "2D-axis-symmetric analytical solution"
        # }
        # END METADATA
        #
        # [...]
        #
        # area_um [um^2]    deform    emodulus [kPa]
        3.75331e+00    5.14496e-03    9.30000e-01
        4.90368e+00    6.72683e-03    9.30000e-01
        6.05279e+00    8.30946e-03    9.30000e-01
        7.20064e+00    9.89298e-03    9.30000e-01
        [...]
    """
    path = pathlib.Path(path).resolve()

    # Parse metadata
    size = path.stat().st_size
    dump = []
    injson = False
    prev_line = ""
    with path.open("r", errors='replace') as fd:
        while True:
            line = fd.readline()
            if fd.tell() == size:
                # something went wrong
                raise ValueError("EOF: Could not parse '{}'!".format(path))
            elif len(line.strip()) == 0:
                # ignore empty lines
                continue
            elif not line.strip().startswith("#"):
                # we are done here
                if prev_line == "":
                    raise ValueError("No column header in '{}'!".format(
                        path))
                break
            elif line.startswith("# BEGIN METADATA"):
                injson = True
                continue
            elif line.startswith("# END METADATA"):
                injson = False
            if injson:
                dump.append(line.strip("#").strip())
            else:
                # remember last line for header
                prev_line = line
    # metadata
    if dump:
        meta = json.loads("\n".join(dump))
    else:
        raise ValueError("No metadata json dump in '{}'!".format(path))
    # header
    feats = []
    units = []
    for hh in prev_line.strip("# ").split("\t"):
        if hh.count(" "):
            ft, un = hh.strip().split(" ")
            un = un.strip("[]")
        else:
            ft = hh
            un = ""
        if ft not in dfn.scalar_feature_names:
            raise ValueError("Feature not known: '{}'".format(ft))
        feats.append(ft)
        units.append(un)
    # data
    with path.open("rb") as lufd:
        data = np.loadtxt(lufd)

    meta["column features"] = feats
    meta["column units"] = units

    # sanity checks
    assert meta["channel_width_unit"] == "um"
    assert meta["flow_rate_unit"] == "uL/s"
    assert meta["fluid_viscosity_unit"] == "mPa s"
    for ft, un in zip(feats, units):
        if ft == "deform":
            assert un == ""
        elif ft == "area_um":
            assert un == "um^2"
        elif ft == "emodulus":
            assert un == "kPa"

    # TODO:
    # - if everything works as expected, add "FEM" to valid methods
    #   and implement in Shape-Out 1/2
    if meta["method"] == "FEM":
        meta["method"] = "numerical"

    return data, meta


def scale_area_um(area_um, channel_width_in, channel_width_out, inplace=False,
                  **kwargs):
    """Perform scale conversion for area_um (linear elastic model)

    The area scales with the characteristic length
    "channel radius" L according to (L'/L)².

    The conversion formula is described in :cite:`Mietke2015`.

    .. versionadded: 0.25.0

    Parameters
    ----------
    area_um: ndarray
        Convex area [µm²]
    channel_width_in: float
        Original channel width [µm]
    channel_width_out: float
        Target channel width [µm]
    inplace: bool
        If True, override input arrays with corrected data
    kwargs:
        not used

    Returns
    -------
    area_um_corr: ndarray
        Scaled area [µm²]
    """
    copy = not inplace
    if issubclass(area_um.dtype.type, np.integer) and inplace:
        raise ValueError("Cannot correct integer `area_um` in-place!")
    area_um_corr = np.array(area_um, copy=copy)

    if channel_width_in != channel_width_out:
        area_um_corr *= (channel_width_out / channel_width_in)**2
    return area_um_corr


def scale_emodulus(emodulus, channel_width_in, channel_width_out,
                   flow_rate_in, flow_rate_out, viscosity_in,
                   viscosity_out, inplace=False):
    """Perform scale conversion for area_um (linear elastic model)

    The conversion formula is described in :cite:`Mietke2015`.

    .. versionadded: 0.25.0

    Parameters
    ----------
    emodulus: ndarray
        Young's Modulus [kPa]
    channel_width_in: float
        Original channel width [µm]
    channel_width_out: float
        Target channel width [µm]
    flow_rate_in: float
        Original flow rate [µL/s]
    flow_rate_out: float
        Target flow rate [µL/s]
    viscosity_in: float
        Original viscosity [mPa*s]
    viscosity_out: float or ndarray
        Target viscosity [mPa*s]; This can be an array
    inplace: bool
        If True, override input arrays with corrected data

    Returns
    -------
    emodulus_corr: ndarray
        Scaled emodulus [kPa]
    """
    copy = not inplace

    emodulus_corr = np.array(emodulus, copy=copy)

    if viscosity_in is not None:
        if isinstance(viscosity_in, np.ndarray):
            raise ValueError("`viscosity_in` must not be an array!")

    if (flow_rate_in != flow_rate_out
        or channel_width_in != channel_width_out
        or (isinstance(viscosity_out, np.ndarray)  # check b4 compare
            or viscosity_in != viscosity_out)):
        emodulus_corr *= (flow_rate_out / flow_rate_in) \
            * (viscosity_out / viscosity_in) \
            * (channel_width_in / channel_width_out)**3

    return emodulus_corr


def scale_feature(feat, data, inplace=False, **scale_kw):
    """Convenience function for scale conversions (linear elastic model)

    This method wraps around all the other scale_* methods and also
    supports deform/circ.

    Parameters
    ----------
    feat: str
        Valid scalar feature name
    data: float or ndarray
        Feature data
    inplace: bool
        If True, override input arrays with corrected data
    **scale_kw:
        Scale keyword arguments for the wrapped methods
    """
    if feat == "area_um":
        return scale_area_um(area_um=data, inplace=inplace, **scale_kw)
    elif feat in ["circ", "deform"]:
        # no units
        return np.array(data, copy=not inplace)
    elif feat == "emodulus":
        return scale_emodulus(emodulus=data, inplace=inplace, **scale_kw)
    elif feat == "volume":
        return scale_volume(volume=data, inplace=inplace, **scale_kw)
    else:
        raise KeyError("No recipe to scale feature '{}'!".format(feat))


def scale_volume(volume, channel_width_in, channel_width_out, inplace=False,
                 **kwargs):
    """Perform scale conversion for volume (linear elastic model)

    The volume scales with the characteristic length
    "channel radius" L according to (L'/L)³.

    .. versionadded: 0.25.0

    Parameters
    ----------
    volume: ndarray
        Volume [µm³]
    channel_width_in: float
        Original channel width [µm]
    channel_width_out: float
        Target channel width [µm]
    inplace: bool
        If True, override input arrays with corrected data
    kwargs:
        not used

    Returns
    -------
    volume_corr: ndarray
        Scaled volume [µm³]
    """
    copy = not inplace
    volume_corr = np.array(volume, copy=copy)

    if channel_width_in != channel_width_out:
        volume_corr *= (channel_width_out / channel_width_in)**3
    return volume_corr


def normalize(data, dmax):
    """Perform normalization in-place for interpolation

    Note that :func:`scipy.interpolate.griddata` has a `rescale`
    option which rescales the data onto the unit cube. For some
    reason this does not work well with LUT data, so we
    just normalize it by dividing by the maximum value.
    """
    data /= dmax
    return data
