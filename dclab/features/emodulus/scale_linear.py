"""Scale conversion applicable to a linear elastic model"""
from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt


def _get_output_array(data: float | npt.NDArray,
                      inplace: bool,
                      name: str) -> npt.NDArray:
    """Return an array for storing the scaled `data`

    If `inplace` is False, a copy of `data` is returned and integer
    data are converted to float.
    """
    if inplace:
        if np.issubdtype(np.asarray(data).dtype, np.integer):
            raise ValueError(f"Cannot correct integer `{name}` in-place!")
        return np.asarray(data)
    data_corr = np.array(data, copy=True)
    if not np.issubdtype(data_corr.dtype, np.inexact):
        data_corr = data_corr.astype(float)
    return data_corr


def convert(area_um: npt.NDArray,
            deform: npt.NDArray,
            channel_width_in: float,
            channel_width_out: float,
            emodulus: npt.NDArray | None = None,
            flow_rate_in: float | None = None,
            flow_rate_out: float | None = None,
            viscosity_in: float | None = None,
            viscosity_out: float | npt.NDArray | None = None,
            inplace: bool = False
            ) -> (
        tuple[npt.NDArray, npt.NDArray] |
        tuple[npt.NDArray, npt.NDArray, npt.NDArray]
):
    """Convert area-deformation-emodulus triplet

    The conversion formula is described in :cite:`Mietke2015`.

    Parameters
    ----------
    area_um
        Convex cell area [µm²]
    deform
        Deformation
    channel_width_in
        Original channel width [µm]
    channel_width_out
        Target channel width [µm]
    emodulus
        Young's Modulus [kPa]
    flow_rate_in
        Original flow rate [µL/s]
    flow_rate_out
        Target flow rate [µL/s]
    viscosity_in
        Original viscosity [mPa*s]
    viscosity_out
        Target viscosity [mPa*s]; this can be an array
    inplace
        If True, overwrite input arrays with corrected data

    Returns
    -------
    area_um_corr
        Corrected cell area [µm²]
    deform_corr
        Deformation (a copy if `inplace` is False)
    emodulus_corr
        Corrected emodulus [kPa]; only returned if `emodulus` is given.

    Notes
    -----
    If only `area_um`, `deform`, `channel_width_in` and
    `channel_width_out` are given, then only the area is
    corrected and returned together with the original deform.
    If `emodulus` is given, then `flow_rate_in`, `flow_rate_out`,
    `viscosity_in`, and `viscosity_out` must be given as well
    (otherwise a ValueError is raised) and the corrected emodulus
    is returned as well.
    """
    warnings.warn("Usage of the `convert` method is deprecated! Please use "
                  "the scale_feature method instead.",
                  DeprecationWarning)
    copy = not inplace

    deform_corr = np.array(deform) if copy else np.asarray(deform)

    if channel_width_in != channel_width_out:
        area_um_corr = scale_area_um(area_um, channel_width_in,
                                     channel_width_out, inplace)
    else:
        area_um_corr = np.array(area_um) if copy else np.asarray(area_um)

    if emodulus is None:
        return area_um_corr, deform_corr
    else:
        missing = [name for name, value in [
            ("flow_rate_in", flow_rate_in),
            ("flow_rate_out", flow_rate_out),
            ("viscosity_in", viscosity_in),
            ("viscosity_out", viscosity_out)] if value is None]
        if missing:
            raise ValueError(f"Correcting `emodulus` requires {missing}!")
        emodulus_corr = scale_emodulus(emodulus, channel_width_in,
                                       channel_width_out, flow_rate_in,
                                       flow_rate_out, viscosity_in,
                                       viscosity_out, inplace)
        return area_um_corr, deform_corr, emodulus_corr


def scale_area_um(area_um: npt.NDArray,
                  channel_width_in: float,
                  channel_width_out: float,
                  inplace: bool = False,
                  **kwargs) -> npt.NDArray:
    """Perform scale conversion for area_um (linear elastic model)

    The area scales with the characteristic length
    "channel radius" L according to (L'/L)².

    The conversion formula is described in :cite:`Mietke2015`.

    .. versionadded:: 0.25.0

    Parameters
    ----------
    area_um
        Convex area [µm²]
    channel_width_in
        Original channel width [µm]
    channel_width_out
        Target channel width [µm]
    inplace
        If True, overwrite input arrays with corrected data
    kwargs
        not used

    Returns
    -------
    area_um_corr
        Scaled area [µm²]
    """
    area_um_corr = _get_output_array(area_um, inplace, "area_um")

    if channel_width_in != channel_width_out:
        area_um_corr *= (channel_width_out / channel_width_in)**2
    return area_um_corr


def scale_emodulus(emodulus: npt.NDArray,
                   channel_width_in: float,
                   channel_width_out: float,
                   flow_rate_in: float,
                   flow_rate_out: float,
                   viscosity_in: float,
                   viscosity_out: float | npt.NDArray,
                   inplace: bool = False) -> npt.NDArray:
    """Perform scale conversion for emodulus (linear elastic model)

    The conversion formula is described in :cite:`Mietke2015`.

    .. versionadded:: 0.25.0

    Parameters
    ----------
    emodulus
        Young's Modulus [kPa]
    channel_width_in
        Original channel width [µm]
    channel_width_out
        Target channel width [µm]
    flow_rate_in
        Original flow rate [µL/s]
    flow_rate_out
        Target flow rate [µL/s]
    viscosity_in
        Original viscosity [mPa*s]
    viscosity_out
        Target viscosity [mPa*s]; this can be an array
    inplace
        If True, overwrite input arrays with corrected data

    Returns
    -------
    emodulus_corr
        Scaled emodulus [kPa]

    Notes
    -----
    If any of the channel widths, flow rates, or viscosities is None,
    `emodulus` is returned unscaled.
    """
    emodulus_corr = _get_output_array(emodulus, inplace, "emodulus")

    if viscosity_in is not None and isinstance(viscosity_in, np.ndarray):
        raise ValueError("`viscosity_in` must not be an array!")

    has_nones = (flow_rate_in is None
                 or flow_rate_out is None
                 or viscosity_in is None
                 or viscosity_out is None
                 or channel_width_in is None
                 or channel_width_out is None
                 )
    has_changes = (flow_rate_in != flow_rate_out
                   or channel_width_in != channel_width_out
                   or (isinstance(viscosity_out, np.ndarray)  # check before
                       or viscosity_in != viscosity_out)
                   )

    if not has_nones and has_changes:
        emodulus_corr *= (flow_rate_out / flow_rate_in) \
            * (viscosity_out / viscosity_in) \
            * (channel_width_in / channel_width_out)**3

    return emodulus_corr


def scale_feature(feat: str,
                  data: float | npt.NDArray,
                  inplace: bool = False,
                  **scale_kw) -> npt.NDArray:
    """Convenience function for scale conversions (linear elastic model)

    This method wraps around all the other scale_* methods and also
    supports deform/circ.

    Parameters
    ----------
    feat
        Valid scalar feature name
    data
        Feature data
    inplace
        If True, overwrite input arrays with corrected data
    **scale_kw
        Scale keyword arguments for the wrapped methods

    Returns
    -------
    data_scaled: ndarray
        Scaled feature data (unchanged for "circ" and "deform")
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
        raise KeyError(f"No recipe to scale feature '{feat}'!")


def scale_volume(volume: npt.NDArray,
                 channel_width_in: float,
                 channel_width_out: float,
                 inplace: bool = False,
                 **kwargs) -> npt.NDArray:
    """Perform scale conversion for volume (linear elastic model)

    The volume scales with the characteristic length
    "channel radius" L according to (L'/L)³.

    .. versionadded:: 0.25.0

    Parameters
    ----------
    volume
        Volume [µm³]
    channel_width_in
        Original channel width [µm]
    channel_width_out
        Target channel width [µm]
    inplace
        If True, overwrite input arrays with corrected data
    kwargs
        not used

    Returns
    -------
    volume_corr
        Scaled volume [µm³]
    """
    volume_corr = _get_output_array(volume, inplace, "volume")

    if channel_width_in != channel_width_out:
        volume_corr *= (channel_width_out / channel_width_in)**3
    return volume_corr
