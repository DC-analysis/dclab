import numpy as np


def get_dry_mass(masked_phase, wavelength, pixel_size,
                 alpha_cell_type=0.19):
    """
    Calculate the dry mass from the phase image.

    Parameters
    ----------
    masked_phase : np.ndarray
        Phase image * event mask. Can be an image stack.
    wavelength : float
        Wavelength of the incident laser illumination in metre.
    pixel_size : float
        Size of a pixel in metre.
    alpha_cell_type
        The alpha constant for the biological sample.
         - 0.19 mL/g for most cells
         - todo: 0.145 ml/g for ... ask kyoo

    Returns
    -------
    dry_mass : float or np.ndarray
        Dry mass of the event. If `masked_phase` is a 3D array (image stack),
        then it will be a np.ndarray of length `masked_phase`.

    Notes
    -----
    Dry Mass of a biological sample can be calculated from Quantitative Phase
    Image data. See reference.

    References
    ----------
    .. [1] Aknoun S et al, Living cell dry mass measurement using quantitative
           phase imaging with quadriwave lateral shearing interferometry: an
           accuracy and sensitivity discussion. J Biomed Opt.
           2015;20(12):126009. doi: 10.1117/1.JBO.20.12.126009.

    """
    if not isinstance(masked_phase, np.ndarray):
        masked_phase = np.asarray(masked_phase)

    summed_phase = np.abs(np.sum(masked_phase, axis=(-1, -2)))

    dry_mass = (summed_phase * (pixel_size ** 2)) * wavelength / (
            2 * np.pi * alpha_cell_type)

    return dry_mass
