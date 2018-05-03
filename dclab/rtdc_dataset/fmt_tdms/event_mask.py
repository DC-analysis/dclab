#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Class for on-the-fly conversion of contours to masks"""
from __future__ import division, print_function, unicode_literals

import numpy as np
import scipy.ndimage as ndi


class MaskColumn(object):
    def __init__(self, rtdc_dataset):
        """Computes mask from contour data"""
        self.contour = rtdc_dataset["contour"]
        self.image = rtdc_dataset["image"]
        self.identifier = self.contour.identifier
        self._rtdc_dataset = rtdc_dataset
        self._shape = None

    def __getitem__(self, idx):
        mask = np.zeros(self.shape, dtype=bool)
        conti = self.contour[idx]
        mask[conti[:, 1], conti[:, 0]] = True
        ndi.morphology.binary_fill_holes(mask, output=mask)
        return mask

    def __len__(self):
        if self.shape != (0, 0):
            lc = len(self.contour)
        else:
            lc = 0
        return lc

    @property
    def shape(self):
        if self._shape is None:
            cfgim = self._rtdc_dataset.config["imaging"]
            if "roi size x" in cfgim and "roi size y" in cfgim:
                # get shape from congig
                self._shape = (cfgim["roi size y"], cfgim["roi size x"])
            elif self.image:
                # get shape from image column
                self._shape = self.image[0].shape
            else:
                # no shape available
                self._shape = (0, 0)
        return self._shape
