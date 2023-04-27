import numpy as np


def get_refractive_index(masked_phase, wavelength, pixel_size,
                         area_um, n_medium=1.337):
    """
    Calculate the refractive index from the phase image.

    Parameters
    ----------
    masked_phase : np.ndarray
        Phase image * event mask. Can be an image stack.
    wavelength : float
        Wavelength of the incident laser illumination in metre.
    pixel_size : float
        Size of a pixel in metre.
    area_um : float
        Area of the event in micrometers. If masked_phase is an image stack,
        this must be a list or array of the same length.
    n_medium : float
        Refractive index of the medium in the channel.

    Returns
    -------
    refractive_index : float or np.ndarray
        Refractive index of the event. If `masked_phase` is a 3D array
        (image stack), then it will be a np.ndarray of length `masked_phase`.

    Notes
    -----
    Refractive index of a biological sample can be calculated from
    Quantitative Phase Image data. See reference below.

    todo: References ask kyoo
    ----------
    .. [1]

    """
    if not isinstance(masked_phase, np.ndarray):
        masked_phase = np.asarray(masked_phase)
    if not isinstance(area_um, np.ndarray):
        area_um = np.asarray(area_um)

    if masked_phase.ndim == 2:
        assert area_um.shape == (1,)
    elif masked_phase.ndim == 3:
        assert masked_phase.shape[0] == area_um.shape[0]

    # area_m = area_um * 1E-6
    summed_phase = np.abs(
        np.sum(masked_phase, axis=(-1, -2))) * (pixel_size ** 2)
    d_n = summed_phase * wavelength * 3 * np.sqrt(
        np.pi / area_um ** 3) / (8 * np.pi)
    n_sample = d_n + n_medium

    return n_sample
