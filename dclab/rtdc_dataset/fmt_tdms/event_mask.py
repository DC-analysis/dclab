"""Class for on-the-fly conversion of contours to masks"""
import numbers

import numpy as np
import scipy.ndimage as ndi


class MaskColumn(object):
    def __init__(self, rtdc_dataset):
        """Computes mask from contour data"""
        self.contour = rtdc_dataset["contour"]
        self.image = rtdc_dataset["image"]
        self.identifier = self.contour.identifier
        self.config = rtdc_dataset.config
        self._shape = None
        self._img_shape_cache = None
        self.dtype = np.bool_

    def __getitem__(self, idx):
        if not isinstance(idx, numbers.Integral):
            raise NotImplementedError(
                "The RTDC_TDMS data handler does not support indexing with "
                "anything else than scalar integers. Please convert your data "
                "to the .rtdc file format!")

        mask = np.zeros(self._img_shape, dtype=bool)
        conti = self.contour[idx]
        mask[conti[:, 1], conti[:, 0]] = True
        ndi.binary_fill_holes(mask, output=mask)
        return mask

    def __len__(self):
        if self._img_shape != (0, 0):
            lc = len(self.contour)
        else:
            lc = 0
        return lc

    @property
    def _img_shape(self):
        if self._img_shape_cache is None:
            """Shape of one event image"""
            cfgim = self.config["imaging"]
            if self.image:
                # get shape from image column
                event_image_shape = self.image.shape[1:]
            elif "roi size x" in cfgim and "roi size y" in cfgim:
                # get shape from config (this is less reliable than getting
                # the shape from the image; there were measurements with
                # wrong config keys)
                event_image_shape = (cfgim["roi size y"], cfgim["roi size x"])
            else:
                # no shape available
                event_image_shape = (0, 0)
            self._img_shape_cache = event_image_shape
        return self._img_shape_cache

    @property
    def shape(self):
        if self._shape is None:
            self._shape = len(self), self._img_shape[0], self._img_shape[1]
        return self._shape
