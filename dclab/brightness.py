#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Computation of mean and standard deviation of grayscale values inside the 
contour for RT-DC measurements
"""
from __future__ import division, print_function, unicode_literals
import numpy as np
import scipy.ndimage



def get_brightness(cont, img, ret_data="avg,sd"):
    """Compute avg and/or std of the event brightness
    
    The event brightness is defined by the gray-scale values of the
    image data within the event contour area. 
    
    Parameters
    ----------
    cont: ndarray or list of ndarrays of shape (N,2)
        A 2D array that holds the contour of an event (in pixels)
        e.g. obtained using `mm.contour` where  `mm` is an instance
        of `RTDCBase`. The first and second columns of `cont`
        correspond to the x- and y-coordinates of the contour.
    img: ndarray or list of ndarrays 
        A 2D array that holds the image in form of gray-scale values of an event    
    ret_data: str
        A comma-separated list of metrices to compute
        - "avg": compute the average
        - "sd": compute the standard deviation
        Selected metrics are returned in alphabetical order.

    Returns
    -------
    bright_avg: float or ndarray of size N
        Average image data within the contour
    bright_std: float or ndarray of size N
        Standard deviation of image data within the contour
    """
    ret_avg = "avg" in ret_data
    ret_std = "sd" in ret_data
    
    assert ret_avg * ret_std != 0, "No metrices selected!"
    
    if not type(img)==list:
        img = [img]
        cont = [cont]
        ret_list = False
    else:
        ret_list = True
    
    msg = '`img` and `cont` must have same size!'
    assert len(img) == len(cont), msg
    
    # Results are stored in a separate array initialized with nans
    if ret_avg:
        avg = np.zeros(len(img), dtype=float) * np.nan
    if ret_std:
        std = np.zeros(len(img), dtype=float) * np.nan

    for ii in range(len(img)):
        imgi = img[ii]
        # Initialize frame mask
        fmi = np.zeros_like(imgi, dtype=bool)
        # Set to true where the contour is
        fmi[cont[ii][:,1], cont[ii][:,0]] = True
        # Fill holes
        scipy.ndimage.morphology.binary_fill_holes(fmi, output=fmi)
        # Assign results
        if ret_avg:
            # Add .5 to the average value to reproduce values of the
            # online analysis. This term is somehow related to how
            # OpenCV performs image calculations differently. Note
            # that there is still a discrepancy (<.1 in grayscale
            # values) due to the different analysis in OpenCV. 
            avg[ii] = np.mean(imgi[fmi])+.5
        if ret_std:
            std[ii] = np.std(imgi[fmi])

    results = []
    # Keep alphabetical order
    if ret_avg:
        results.append(avg)
    if ret_std:
        results.append(std)
    
    if not ret_list:
        results = [ r[0] for r in results ]
     
    return results

