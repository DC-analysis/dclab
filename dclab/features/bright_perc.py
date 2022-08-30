"""
Computation of the 10th and 90th percentile of grayscale values inside the
RT-DC event image mask with background-correction taken into account.
"""
import numpy as np


def get_bright_perc(mask, image, image_bg):
    """Compute 10th and 90th percentile of the bg-corrected event brightness

    The background-corrected event brightness is defined by the
    gray-scale values of the background-corrected image data
    within the event mask area.

    Parameters
    ----------
    mask: ndarray or list of ndarrays of shape (M,N) and dtype bool
        The mask values, True where the event is located in `image`.
    image: ndarray or list of ndarrays of shape (M,N)
        A 2D array that holds the image in form of grayscale values
        of an event.
    image_bg: ndarray or list of ndarrays of shape (M,N)
        A 2D array that holds the background image for the same event.

    Returns
    -------
    bright_perc_10: float or ndarray of size N
        10th percentile of brightness
    bright_perc_10: float or ndarray of size N
        90th percentile of brightness
    """
    if isinstance(mask, np.ndarray) and len(mask.shape) == 2:
        # We have a single image
        image_bg = [image_bg]
        image = [image]
        mask = [mask]
        ret_list = False
    else:
        ret_list = True

    length = min(len(mask), len(image), len(image_bg))

    p10 = np.zeros(length, dtype=float) * np.nan
    p90 = np.zeros(length, dtype=float) * np.nan

    for ii in range(length):
        # cast to integer before subtraction
        imgi = np.array(image[ii], dtype=int) - image_bg[ii]
        mski = mask[ii]
        # Assign results
        p10[ii], p90[ii] = np.percentile(imgi[mski], q=[10, 90])

    if ret_list:
        return p10, p90
    else:
        return p10[0], p90[0]
