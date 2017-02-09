#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Class for efficiently handling contour data
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
import os


class ContourColumn(object):
    def __init__(self, rtdc_dataset):
        """A wrapper for ContourData that takes into account event offsets
        
        Event offsets appear when the first event that is recorded in the
        tdms files does not have a corresponding contour in the contour
        text file.
        """
        fname = self.find_contour_file(rtdc_dataset)
        if fname is not None:
            self._contour_data = ContourData(fname)
            self._initialized = False
        else:
            self._contour_data = []
            # prevent `determine_offset` to be called
            self._initialized = True
        self._rtdc_dataset = rtdc_dataset
        self.event_offset = 0
    
    
    def __getitem__(self, idx):
        if not self._initialized:
            self.determine_offset()
        idnew = idx-self.event_offset
        if idnew < 0:
            cdata = np.zeros((2,2), dtype=int)
        else:
            cdata = self._contour_data[idnew]
        return cdata


    def __len__(self):
        length = len(self._contour_data)
        if length:
            length += self.event_offset 
        return length


    def determine_offset(self):
        """Determines the offset of the contours w.r.t. other data columns
        
        
        Notes
        -----
        - the "frame" column of `self._rtdc_dataset` is compared to
          the first contour in the contour text file to determine an
          offset by one event
        - modifies the property `event_offset` and sets `_initialized`
          to `True`
        """
        # In case of regular RTDC, the first contour is
        # missing. In case of fRTDC, it is there, so we
        # might have an offset. We find out if the first
        # contour frame is missing by comparing it to
        # the "frame" column of the rtdc data set.
        fref = self._contour_data.get_frame(0)
        f0 = self._rtdc_dataset.frame[0]
        f1 = self._rtdc_dataset.frame[1]
        if fref == f0:
            self.event_offset = 0
        elif fref == f1:
            self.event_offset = 1
        else:
            raise IndexError("Contour data has unknown offset!")
        self._initialized = True


    @staticmethod
    def find_contour_file(rtdc_dataset):
        """Tries to find a contour file that belongs to an RTDC data set
        
        Returns None if no contour file is found.
        """
        cfile = None
        if os.path.exists(rtdc_dataset.fdir):
            for f in os.listdir(rtdc_dataset.fdir):
                if (f.endswith("_contours.txt") and
                    f.startswith(rtdc_dataset.name[:2])):
                    cfile = os.path.join(rtdc_dataset.fdir, f)
                    break
        return cfile



class ContourData(object):
    def __init__(self, fname):
        """Access an MX_contour.txt as a dictionary
        
        Initialize this class with a *_contour.txt file.
        The individual contours can be accessed like a
        list (enumerated from 0 on).
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
        """Open and index the contour file
        
        This function populates the internal list of contours
        as strings which will be available as `self.data`.
        """
        with open(self.filename) as fd:
            data = fd.read()
            
        ident = "Contour in frame"
        self._data = data.split(ident)[1:]
        self._initialized = True
        

    @property
    def data(self):
        """Access self.data
        If `self._index_file` has not been computed before, this
        property will cause it to do so.
        """
        if not self._initialized:
            self._index_file()
        return self._data


    def get_frame(self, idx):
        """Return the frame number of a contour"""
        cont = self.data[idx]
        frame = int(cont.strip().split(" ", 1)[0])
        return frame
        
        
        