#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Handling fluorescence trace data"""
from __future__ import division, print_function, unicode_literals

import pathlib

from nptdms import TdmsFile
import numpy as np

from ... import definitions as dfn

from . import naming


class TraceColumn(object):
    def __init__(self, rtdc_dataset):
        """Prepares everything but does not load the trace data yet

        The trace data is loaded when __getitem__, __len__, or __iter__
        are called. This saves time and memory when the trace data is
        not needed at all, e.g. for batch processing with Shape-Out.
        """
        self._trace = None
        self.mname = rtdc_dataset.path
        self.identifier = self.mname

    def __getitem__(self, trace_key):
        if trace_key not in dfn.FLUOR_TRACES:
            msg = "Unknown fluorescence trace key: {}".format(trace_key)
            raise ValueError(msg)
        return self.trace.__getitem__(trace_key)

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

        if tname is None:
            rep = "No trace data available!"
        else:
            rep = "Fluorescence trace data from file {}, <{}>".format(tname,
                                                                      addstr)
        return rep

    def keys(self):
        return self.trace.keys()

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
        elif tname.suffix == ".tdms":
            # Again load the measurement tdms file.
            # This might increase memory usage, but it is cleaner
            # when looking at code structure.
            mdata = TdmsFile(str(mname))
            sampleids = mdata.object("Cell Track", "FL1index").data

            # Load the trace data. The traces file is usually larger than the
            # measurement file.
            tdata = TdmsFile(str(tname))
            for trace_key in dfn.FLUOR_TRACES:
                group, ch = naming.tr_data_map[trace_key]
                try:
                    trdat = tdata.object(group, ch).data
                except KeyError:
                    pass
                else:
                    if trdat is not None and trdat.size != 0:
                        # Only add trace if there is actual data.
                        # Split only needs the position of the sections,
                        # so we remove the first (0) index.
                        trace[trace_key] = np.split(trdat, sampleids[1:])
        return trace

    @staticmethod
    def find_trace_file(mname):
        """Tries to find the traces tdms file name

        Returns None if no trace file is found.
        """
        mname = pathlib.Path(mname)
        tname = None

        if mname.exists():
            cand = mname.with_name(mname.name[:-5] + "_traces.tdms")
            if cand.exists():
                tname = cand

        return tname
