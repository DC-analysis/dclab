#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This library contains classes and methods for the analysis
of real-time deformability cytometry (RT-DC) data sets.
"""
from __future__ import division, print_function

import codecs
import copy
import hashlib
from nptdms import TdmsFile
import numpy as np
import os
from scipy.stats import norm, gaussian_kde
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import time
    
import warnings

# Definitions
from . import definitions as dfn
from ._version import version as __version__
from .rtdc_dataset import RTDC_DataSet, GetProjectNameFromPath
from .polygon_filter import PolygonFilter
        
def crop_linear_data(data, xmin, xmax, ymin, ymax):
    """ Crop plotting data.
    
    Crops plotting data of monotonous function and linearly interpolates
    values at end of interval.
    
    Paramters
    ---------
    data : ndarray of shape (N,2)
        The data to be filtered in x and y.
    xmin : float
        minimum value for data[:,0]
    xmax : float
        maximum value for data[:,0]
    ymin : float
        minimum value for data[:,1]
    ymax : float
        maximum value for data[:,1]    
    
    
    Returns
    -------
    ndarray of shape (M,2), M<=N
    
    Notes
    -----
    `data` must be monotonically increasing in x and y.
    
    """
    # TODO:
    # Detect re-entering of curves into plotting area
    x = data[:,0].copy()
    y = data[:,1].copy()
    
    # Filter xmin
    if np.sum(x<xmin) > 0:
        idxmin = np.sum(x<xmin)-1
        xnew = x[idxmin:].copy()
        ynew = y[idxmin:].copy()
        xnew[0] = xmin
        ynew[0] = np.interp(xmin, x, y)
        x = xnew
        y = ynew


    # Filter ymax
    if np.sum(y>ymax) > 0:
        idymax = len(y)-np.sum(y>ymax)+1
        xnew = x[:idymax].copy()
        ynew = y[:idymax].copy()
        ynew[-1] = ymax
        xnew[-1] = np.interp(ymax, y, x)
        x = xnew
        y = ynew
        

    # Filter xmax
    if np.sum(x>xmax) > 0:
        idxmax = len(y)-np.sum(x>xmax)+1
        xnew = x[:idxmax].copy()
        ynew = y[:idxmax].copy()
        xnew[-1] = xmax
        ynew[-1] = np.interp(xmax, x, y)
        x = xnew
        y = ynew
        
    # Filter ymin
    if np.sum(y<ymin) > 0:
        idymin = np.sum(y<ymin)-1
        xnew = x[idymin:].copy()
        ynew = y[idymin:].copy()
        ynew[0] = ymin
        xnew[0] = np.interp(ymin, y, x)
        x = xnew
        y = ynew
    
    newdata = np.zeros((len(x),2))
    newdata[:,0] = x
    newdata[:,1] = y

    return newdata

   
def GetTDMSFiles(directory):
    """ Recursively find projects based on '.tdms' file endings
    
    Searches the `directory` recursively for '.tdms' project files.
    Returns a list of files.
    
    If the callback function is defined, it will be called for each
    directory.
    """
    directory = os.path.realpath(directory)
    tdmslist = list()
    for root, _dirs, files in os.walk(directory):
        for f in files:
            # Philipp:
            # Exclude traces files of fRT-DC setup
            if (f.endswith(".tdms") and (not f.endswith("_traces.tdms"))):
                tdmslist.append(os.path.realpath(os.path.join(root,f)))
    tdmslist.sort()
    return tdmslist


def _get_data_path():
    return os.path.realpath(os.path.dirname(__file__))

