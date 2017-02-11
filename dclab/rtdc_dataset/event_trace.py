#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Class for handling fluorescence trace data
"""
from __future__ import division, print_function, unicode_literals

import nptdms
import numpy as np
import os

from .. import definitions as dfn


class TraceColumn(object):
    def __init__(self, rtdc_dataset):
        """Prepares everything but does not load the trace data yet
        
        The trace data is loaded when __getitem__, __len__, or __iter__
        are called. This saves time and memory when the trace data is
        not needed at all, e.g. for batch processing with ShapeOut. 
        """
        self._trace = None
        self.mname = rtdc_dataset.tdms_filename
        

    def __getitem__(self, ch):
        return self.trace.__getitem__(ch)


    def __len__(self):
        return self.trace.__len__()


    def __iter__(self):
        return self.trace.__iter__()


    def __repr__(self):
        tname = TraceColumn.find_trace_file(self.mname)
        if self._trace is None:
            addstr = "not loaded into memory"
        else:
            addstr = "loaded into memory"
        return ("Fluorescence trace data from file {}, <{}>".format(tname,
                                                                    addstr))


    @property
    def trace(self):
        """Initializes the trace data"""
        if self._trace is None:
            self._trace = self.load_trace(self.mname)
        return self._trace


    @staticmethod
    def load_trace(mname):
        """Loads the traces and returns them as a dictionary
        
        Currently, only loading traces from tdms files is supported.
        This forces us to load the full tdms file into memory which
        takes some time.
        """
        tname = TraceColumn.find_trace_file(mname)
        
        # Initialize empty trace dictionary
        trace = {}

        if tname is None:
            pass
        elif tname.endswith(".tdms"):
            # Again load the measurement tdms file.
            # This might increase memory usage, but it is cleaner
            # when looking at code structure.
            mdata = nptdms.TdmsFile(mname)
            sampleids = mdata.object("Cell Track", "FL1index").data
            
            # Load the trace data. The traces file is usually larger than the
            # measurement file.
            tdata = nptdms.TdmsFile(tname)
            for group, ch in dfn.tr_data:
                try:
                    trdat = tdata.object(group, ch).data
                except KeyError:
                    pass
                else:
                    if trdat is not None:
                        # Only add trace if there is actual data.
                        # Split only needs the position of the sections,
                        # so we remove the first (0) index.
                        trace[ch] = np.split(trdat, sampleids[1:])
        return trace
        

    @staticmethod
    def find_trace_file(mname):
        """Tries to find the traces tdms file name
        
        Returns None if no trace file is found.
        """
        tname = None
        
        if os.path.exists(mname):
            cand = mname[:-5]+"_traces.tdms"
            if os.path.exists(cand):
                tname = cand
            
        return tname
