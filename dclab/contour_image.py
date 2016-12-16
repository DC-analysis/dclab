#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Class for efficiently handling the contour data
"""
from __future__ import division, print_function, unicode_literals

import numpy as np

class ContourImage(object):
    def __init__(self, fname):
        """Access an MX_contour.txt like a dictionary
        
        The frame is the key
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


    def __len__(self):
        return len(self.data)
    

    def _index_file(self):
        """Initially index the contour file
        
        This function populates the internal frame dictionary.
        """
        with open(self.filename) as fd:
            data = fd.read()
            
        ident = "Contour in frame"
        self._data = data.split(ident)[1:]
        self._initialized = True


    @property
    def data(self):
        if not self._initialized:
            self._index_file()
        return self._data
