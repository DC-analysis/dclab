#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Statistics computation for RT-DC dataset instances
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
import traceback as tb
import warnings

from . import definitions as dfn


class Statistics(object):
    available_methods = {}

    def __init__(self, name, method, req_feature=False):
        """ A helper class for statistics.
        
        All statistical methods are registered in the dictionary
        `Statistics.available_methods`.
        """
        self.method = method
        self.name = name
        self.req_feature = req_feature
        Statistics.available_methods[name] = self

    def get_feature(self, rtdc_ds, axis):
        axis = axis.lower()
        if rtdc_ds.config["filtering"]["enable filters"]:
            x = rtdc_ds[axis][rtdc_ds._filter]
        else:
            x = rtdc_ds[axis]
        bad = np.isnan(x)^np.isinf(x)
        xout = x[~bad]
        return xout
    
    def get_data(self, kwargs):
        if "rtdc_ds" not in kwargs:
            raise ValueError("Keyword argument 'rtdc_ds' missing.")
        
        rtdc_ds = kwargs["rtdc_ds"]

        if self.req_feature:
            if "feature" not in kwargs:
                raise ValueError("Keyword argument 'feature' missing.")
            return self.get_feature(rtdc_ds, kwargs["feature"])
        else:
            return rtdc_ds

    def __call__(self, **kwargs):
        data = self.get_data(kwargs)
        if len(data) == 0:
            result = np.nan
        else:
            try:
                result = self.method(data)
            except:
                exc = tb.format_exc().replace("\n", "\n    | ")
                warnings.warn("Failed to compute {} for {}: {}".format(
                              self.name, kwargs["rtdc_ds"].title, exc))
                result = np.nan
        return result


def flow_rate(mm):
    conf = mm.config["setup"]
    if "flow rate" in conf:
        return conf["flow rate"]
    else:
        return np.nan

    
def get_statistics(rtdc_ds, methods=None, features=None):
    """
    Parameters
    ----------
    rtdc_ds : instance of `dclab.rtdc_dataset.RTDCBase`.
        The data set for which to compute the statistics.
    methods : list of str or None
        The methods wih which to compute the statistics.
        The list of available methods is given with
        `dclab.statistics.Statistics.available_methods.keys()`
        If set to `None`, statistics for all methods are computed.
    features : list of str
        Feature name identifiers are defined in
        `dclab.definitions.feature_names`.
        If set to `None`, statistics for all axes are computed. 
    
    Returns
    -------
    header : list of str
        The header (feature + method names) of the computed statistics.
    values : list of float
        The computed statistics.
    """
    if methods is None:
        cls = list(Statistics.available_methods.keys())
        # sort the features in a usable way
        me1 = [ m for m in cls if not Statistics.available_methods[m].req_feature ]
        me2 = [ m for m in cls if Statistics.available_methods[m].req_feature ]
        methods = me1 + me2

    if features is None:
        features = dfn.feature_names
    else:
        features = [a.lower() for a in features]
    
    header = []
    values = []

    # To make sure that all methods are computed for each feature in a block,
    # we loop over all features. It would be easier to loop over the methods,
    # but the resulting statistics would not be human-friendly.
    for ft in features:
        for mt in methods:
            meth = Statistics.available_methods[mt]
            if meth.req_feature:
                if ft in rtdc_ds:
                    values.append(meth(rtdc_ds=rtdc_ds, feature=ft))
                else:
                    values.append(np.nan)
                header.append(" ".join([mt, dfn.feature_name2label[ft]]))
            else:
                # Prevent multiple entries of this method.
                if not header.count(mt):
                    values.append(meth(rtdc_ds=rtdc_ds))
                    header.append(mt)

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
Statistics(name="Mean",   req_feature=True, method=np.average)
Statistics(name="Median", req_feature=True, method=np.median)
Statistics(name="Mode",   req_feature=True, method=mode)
Statistics(name="SD",     req_feature=True, method=np.std)
# Methods that work on RTDCBase
Statistics(name="Events",
           method=lambda mm: np.sum(mm._filter))
Statistics(name="%-gated",
           method=lambda mm: np.average(mm._filter)*100)
Statistics(name="Flow rate",
           method=lambda mm: flow_rate(mm))
