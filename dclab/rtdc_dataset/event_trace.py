#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Class for efficiently handling fluorescence trace data
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
import os

# TODO:
# - make this work
# - make traces available as dictionaries
# - replace rtdc_dataset.traces with rtdc_dataset.trace


class TraceColumn(dict):
    def __init__(self, rtdc_dataset):
        """Indexes the fluorescence traces.
        """
        fname = self.find_traces_file(rtdc_dataset)
        if fname is not None:
            self._trace_data = TraceData(fname)
        else:
            self._trace_data = {}
        

    def __getitem__(self, ch):
        """Get a trace channel"""
        if ch in self._trace_data:
            chan = self._trace_data[ch]
        return chan


    def __len__(self):
        length = len(self._image_data)
        if length:
            length += self.event_offset
        return length


    @staticmethod
    def find_traces_file(rtdc_dataset):
        """Tries to find the traces tdms file name
        
        Returns None if no video file is found.
        """
        tmds = rtdc_dataset.tdms_filename
        if os.path.exists(tmds):
            traces_filename = tmds[:-5]+"_traces.tdms"
        else:
            traces_filename = None
        
        return traces_filename



class TraceData(object):
    def __init__(self, fname):
        """Access a _traces.tdms file as a dictionary
        
        Initialize this class with a *_traces.tdms file.
        The individual traces can be accessed like a
        list (enumerated from 0 on).
        """
        self._initialized = False
        self.filename=fname

