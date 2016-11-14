#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Statistics computation for RTDC_DataSet instances
"""
from __future__ import division, print_function

import numpy as np
import traceback as tb
import warnings

from . import definitions as dfn


class Statistics(object):
    available_methods = {}

    def __init__(self, name, method, req_axis=False):
        """ A helper class for statistics.
        
        All statistical methods are registered in the dictionary
        `Statistics.available_methods`.
        """
        self.method=method
        self.name=name
        self.req_axis=req_axis
        Statistics.available_methods[name]=self

    def get_column(self, rtdc_ds, axis):
        if rtdc_ds.Configuration["Filtering"]["Enable Filters"]:
            x = getattr(rtdc_ds, dfn.cfgmaprev[axis])[rtdc_ds._filter]
        else:
            x = getattr(rtdc_ds, dfn.cfgmaprev[axis])
        return x
    
    def get_data(self, kwargs):
        assert "rtdc_ds" in kwargs, "Keyword argument 'rtdc_ds' missing."
        rtdc_ds = kwargs["rtdc_ds"]

        if self.req_axis:
            assert "axis" in kwargs, "Keyword argument 'axis' missing."
            return self.get_column(rtdc_ds, kwargs["axis"])
        else:
            return rtdc_ds

    def __call__(self, **kwargs):
        data = self.get_data(kwargs)
        try:
            result = self.method(data)
        except:
            warnings.warn("Failed to compute {} for {}: {}".format(
                          self.name, kwargs["rtdc_ds"].title, tb.format_exc()))
            result = np.nan
        return result


def flow_rate(mm):
    conf = mm.Configuration["General"]
    if "Flow Rate [ul/s]" in conf:
        return conf["Flow Rate [ul/s]"]
    else:
        return np.nan

    
def get_statistics(rtdc_ds, columns=None, axes=None):
    """
    Parameters
    ----------
    rtdc_ds : instance of `dclab.RTDC_DataSet`.
        The data set for which to compute the statistics.
    columns : list of str or None
        The columns for which to compute the statistics.
        The list of available methods is given with
        `dclab.statistics.Statistics.available_methods.keys()`
        If set to `None`, statistics for all columns are computed.
    axes : list of str
        Column name identifiers are defined in
        `dclab.definitions.uid`.
        If set to `None`, statistics for all axes are computed. 
    
    Returns
    -------
    header : list of str
        The header (column names) of the computed statistics.
    values : list of float
        The computed statistics.
    """
    if columns is None:
        cls = list(Statistics.available_methods.keys())
        # sort the columns in a usable way
        c1 = [ c for c in cls if not Statistics.available_methods[c].req_axis ]
        c2 = [ c for c in cls if Statistics.available_methods[c].req_axis ]
        columns = c1+c2

    if axes is None:
        axes = dfn.uid
    
    header = []
    values = []

    # To make sure that all columns are computed for each axis in a block,
    # we loop over all axes. It would be easier to loop over the columns,
    # but the resulting statistics would be human-friendly.
    for ax in axes:
        for c in columns:
            meth = Statistics.available_methods[c]
            if meth.req_axis:
                values.append(meth(rtdc_ds=rtdc_ds, axis=ax))
                header.append(" ".join([c, dfn.axlabels[ax]]))
            else:
                # Prevent multiple entries of this column.
                if not header.count(c):
                    values.append(meth(rtdc_ds=rtdc_ds))
                    header.append(c)

    return header, values


def mode(data):
    """ Compute an intelligent value for the mode
    
    The most common value in experimental is not very useful if there
    are a lot of digits after the comma. This method approaches this
    issue by rounding to bin size that is determined by the
    Freedman–Diaconis rule.
    
    Parameters
    ----------
    data : 1d ndarray
        The data for which the mode should be computed.
    
    Returns
    -------
    mode : float
        The mode computed with the Freedman-Diaconis rule.
    """
    # size
    n = data.shape[0]
    # interquartile range
    iqr = np.percentile(data, 75)-np.percentile(data, 25)
    # Freedman–Diaconis
    bin_size = 2 * iqr / n**(1/3)
    
    if bin_size == 0:
        return np.nan
    
    # Add bin_size/2, because we want the center of the bin and
    # not the left corner of the bin.
    databin = np.round(data/bin_size)*bin_size + bin_size/2
    u, indices = np.unique(databin, return_inverse=True)
    mode = u[np.argmax(np.bincount(indices))]
    
    return mode


## Register all the methods
# Methods that require an axis
Statistics(name="Mean",   req_axis=True, method=np.average)
Statistics(name="Median", req_axis=True, method=np.median)
Statistics(name="Mode",   req_axis=True, method=mode)
Statistics(name="SD",     req_axis=True, method=np.std)
# Methods that work on an RTDC_DataSet
Statistics(name="Events",
           method=lambda mm: np.sum(mm._filter))
Statistics(name="%-gated",
           method=lambda mm: np.average(mm._filter)*100)
Statistics(name="Flow rate",
           method=lambda mm: flow_rate(mm))
