"""Handling fluorescence trace data"""
import pathlib
import warnings

from nptdms import TdmsFile
import numpy as np

from ... import definitions as dfn

from . import naming
from .exc import InvalidTDMSFileFormatError, MultipleSamplesPerEventFound


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
            try:
                sampleids = mdata["Cell Track"]["FL1index"].data
            except KeyError:
                raise InvalidTDMSFileFormatError(
                    "No 'FL1index' column in '{}'!".format(tname))

            # Check that sample IDs are always incremented with same
            # sample size.
            samples_per_event = np.unique(np.diff(sampleids))
            if len(samples_per_event) > 1:
                # This means the length of the fluorescence trace is not
                # a constant. According to Philipp, this means the trace
                # cannot be used.
                warnings.warn("Ignoring trace data of '{}' ".format(tname)
                              + "due to multiple values for samples per "
                              + "event: {}".format(samples_per_event),
                              MultipleSamplesPerEventFound)
            else:
                # Load the trace data. The traces file is usually larger than
                # the measurement file.
                tdata = TdmsFile(str(tname))
                for trace_key in dfn.FLUOR_TRACES:
                    group, ch = naming.tr_data_map[trace_key]
                    try:
                        trdat = tdata[group][ch].data
                    except KeyError:
                        pass
                    else:
                        if trdat is not None and trdat.size != 0:
                            # Split the input trace data into equally-spaced
                            # sections (we already tested that sampleids is
                            # equally-spaced).
                            spe = sampleids[1] - sampleids[0]
                            trace[trace_key] = np.split(trdat, trdat.size//spe)
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
