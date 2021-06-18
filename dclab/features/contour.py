"""Computation of event contour from event mask"""
import numbers

import numpy as np

# equivalent to
# from skimage.measure import find_contours
from ..external.skimage.measure import find_contours


class NoValidContourFoundError(BaseException):
    pass


class LazyContourList(object):
    def __init__(self, masks):
        """A list-like object that computes contours upon indexing"""
        self.masks = masks
        self.contours = [None] * len(masks)
        #: used for hashing in ancillary features
        self.identifier = str(masks[0][:].tobytes())

    def __getitem__(self, idx):
        """Compute contour(s) if not already in self.contours"""
        if not isinstance(idx, numbers.Integral):
            # slicing!
            indices = np.arange(len(self))[idx]
            output = []
            # populate the output list
            for evid in indices:
                output.append(self.__getitem__(evid))
            return output
        else:
            if self.contours[idx] is None:
                try:
                    self.contours[idx] = get_contour(self.masks[idx])
                except BaseException as e:
                    e.args = ("Event {}, {}".format(idx, e.args[0]),)
                    raise
            return self.contours[idx]

    def __len__(self):
        return len(self.masks)


def get_contour(mask):
    """Compute the image contour from a mask

    The contour is computed in a very inefficient way using scikit-image
    and a conversion of float coordinates to pixel coordinates.

    Parameters
    ----------
    mask: binary ndarray of shape (M,N) or (K,M,N)
        The mask outlining the pixel positions of the event.
        If a 3d array is given, then `K` indexes the individual
        contours.

    Returns
    -------
    cont: ndarray or list of K ndarrays of shape (J,2)
        A 2D array that holds the contour of an event (in pixels)
        e.g. obtained using `mm.contour` where  `mm` is an instance
        of `RTDCBase`. The first and second columns of `cont`
        correspond to the x- and y-coordinates of the contour.
    """
    if isinstance(mask, np.ndarray) and len(mask.shape) == 2:
        mask = [mask]
        ret_list = False
    else:
        ret_list = True
    contours = []

    for mi in mask:
        conts = find_contours(mi.transpose(),
                              level=.9999,
                              positive_orientation="low",
                              fully_connected="high")
        # get the longest contour
        c0 = sorted(conts, key=lambda x: len(x))[-1]
        # round all coordinates to pixel values
        c1 = np.asarray(np.round(c0), int)
        # remove duplicates
        c2 = remove_duplicates(c1)
        if len(c2) == 0:
            raise NoValidContourFoundError("No contour found!")
        contours.append(c2)
    if ret_list:
        return contours
    else:
        return contours[0]


def get_contour_lazily(mask):
    """Like :func:`get_contour`, but computes contours on demand

    Parameters
    ----------
    mask: binary ndarray of shape (M,N) or (K,M,N)
        The mask outlining the pixel positions of the event.
        If a 3d array is given, then `K` indexes the individual
        contours.

    Returns
    -------
    cont: ndarray or LazyContourList of K ndarrays of shape (J,2)
        A 2D array that holds the contour of an event (in pixels)
        e.g. obtained using `mm.contour` where  `mm` is an instance
        of `RTDCBase`. The first and second columns of `cont`
        correspond to the x- and y-coordinates of the contour.
    """
    if isinstance(mask, np.ndarray) and len(mask.shape) == 2:
        # same behavior as `get_contour`
        cont = get_contour(mask=mask)
    else:
        cont = LazyContourList(masks=mask)
    return cont


def remove_duplicates(cont):
    out = []
    for ii in range(len(cont)):
        if np.all(cont[ii] == cont[ii - 1]):
            pass
        else:
            out.append(cont[ii])
    return np.array(out)
