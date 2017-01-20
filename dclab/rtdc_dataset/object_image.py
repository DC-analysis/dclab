#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Class for efficiently handling image/video data
"""
from __future__ import division, print_function, unicode_literals

import numpy as np

class ImageMap(object):
    def __init__(self, fname):
        """Access a video file of an RT-DC data set
        
        Initialize this class with a video file.
        """
        self._initialized = False
        self.filename=fname


    def __getitem__(self, idx):
        cont = self.data[idx]
        cont = cont.strip()
        cont = cont.splitlines()
        if len(cont) > 1:
            _frame = int(cont.pop(0))
            cont = [ np.fromstring(c.strip("()"), sep=",") for c in cont ]
            cont = np.array(cont, dtype=np.uint8)
            return cont
