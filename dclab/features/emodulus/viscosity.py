"""Viscosity computation for various media"""
import warnings

import numpy as np

from ...warn import PipelineWarning


#: Dictionary with different names for one medium
SAME_MEDIA = {
    "0.49% MC-PBS": ["0.49% MC-PBS",
                     "0.5% MC-PBS",
                     "0.50% MC-PBS",
                     "CellCarrier",
                     ],
    "0.59% MC-PBS": ["0.59% MC-PBS",
                     "0.6% MC-PBS",
                     "0.60% MC-PBS",
                     "CellCarrier B",
                     "CellCarrierB",
                     ],
    "0.83% MC-PBS": ["0.83% MC-PBS",
                     "0.8% MC-PBS",
                     "0.80% MC-PBS"],
    "water": ["water"],
}

#: Many media names are actually shorthand for one medium
ALIAS_MEDIA = {}
for key in SAME_MEDIA:
    for item in SAME_MEDIA[key]:
        ALIAS_MEDIA[item] = key
        ALIAS_MEDIA[item.lower()] = key  # also support all-lower case

#: Media for which computation of viscosity is defined (has duplicate entries)
KNOWN_MEDIA = sorted(ALIAS_MEDIA.keys())


class TemperatureOutOfRangeWarning(PipelineWarning):
    pass


def get_viscosity(medium="CellCarrier", channel_width=20.0, flow_rate=0.16,
                  temperature=23.0):
    """Returns the viscosity for RT-DC-specific media

    Media that are not pure (e.g. ketchup or polymer solutions)
    often exhibit a non-linear relationship between shear rate
    (determined by the velocity profile) and shear stress
    (determined by pressure differences). If the shear stress
    grows non-linearly with the shear rate resulting in a slope
    in log-log space that is less than one, then we are talking about
    shear thinning. The viscosity is not a constant anymore (as it
    is e.g. for water). At higher flow rates, the viscosity becomes
    smaller, following a power law. Christoph Herold characterized
    shear thinning for the CellCarrier media :cite:`Herold2017`.
    The resulting formulae for computing the viscosities of these
    media at different channel widths, flow rates, and temperatures,
    are implemented here.

    Parameters
    ----------
    medium: str
        The medium to compute the viscosity for; Valid values
        are defined in :const:`KNOWN_MEDIA`.
    channel_width: float
        The channel width in µm
    flow_rate: float
        Flow rate in µL/s
    temperature: float or ndarray
        Temperature in °C

    Returns
    -------
    viscosity: float or ndarray
        Viscosity in mPa*s

    Notes
    -----
    - CellCarrier and CellCarrier B media are optimized for
      RT-DC measurements.
    - Values for the viscosity of water are computed using
      equation (15) from :cite:`Kestin_1978`.
    - A :class:`TemperatureOutOfRangeWarning` is issued if the
      input temperature range exceeds the temperature ranges given
      by :cite:`Herold2017` and :cite:`Kestin_1978`.
    """
    # also support lower-case media and a space before the "B"
    valmed = [v.lower() for v in KNOWN_MEDIA + ["CellCarrier B"]]
    medium = medium.lower()
    if medium not in valmed:
        raise ValueError("Invalid medium: {}".format(medium))

    # convert flow_rate from µL/s to m³/s
    # convert channel_width from µm to m
    term1 = 1.1856 * 6 * flow_rate * 1e-9 / (channel_width * 1e-6)**3 * 2 / 3

    if medium == "cellcarrier":
        temp_corr = (temperature / 23.2)**-0.866
        term2 = 0.6771 / 0.5928 + 0.2121 / (0.5928 * 0.677)
        eta = 0.179 * (term1 * term2)**(0.677 - 1) * temp_corr * 1e3
        if np.min(temperature) < 16 or np.max(temperature) > 26:
            # see figure (9) in Herold arXiv:1704.00572 (2017)
            warnings.warn("For CellCarrier, the temperature should be in "
                          + "[18, 26] degC! Got min/max of "
                          + "[{:.1f}, {:.1f}] degC.".format(
                              np.min(temperature), np.max(temperature)),
                          TemperatureOutOfRangeWarning)
    elif medium in ["cellcarrierb", "cellcarrier b"]:
        temp_corr = (temperature / 23.6)**-0.866
        term2 = 0.6771 / 0.5928 + 0.2121 / (0.5928 * 0.634)
        eta = 0.360 * (term1 * term2)**(0.634 - 1) * temp_corr * 1e3
        if np.min(temperature) < 16 or np.max(temperature) > 26:
            # see figure (9) in Herold arXiv:1704.00572 (2017)
            warnings.warn("For CellCarrier B, the temperature should be in "
                          + "[18, 26] degC! Got min/max of "
                          + "[{:.1f}, {:.1f}] degC.".format(
                              np.min(temperature), np.max(temperature)),
                          TemperatureOutOfRangeWarning)
    elif medium == "water":
        if np.min(temperature) < 0 or np.max(temperature) > 40:
            # see equation (15) in Kestin et al, J. Phys. Chem. 7(3) 1978
            warnings.warn("For water, the temperature should be in [0, 40] "
                          + "degC! Got min/max of "
                          + "[{:.1f}, {:.1f}] degC.".format(
                              np.min(temperature), np.max(temperature)),
                          TemperatureOutOfRangeWarning)
        eta0 = 1.002  # [mPa s]
        right = (20-temperature) / (temperature + 96) \
            * (+ 1.2364
               - 1.37e-3 * (20 - temperature)
               + 5.7e-6 * (20 - temperature)**2
               )
        eta = eta0 * 10**right
    return eta
