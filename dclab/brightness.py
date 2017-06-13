#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Computation of mean and standard deviation of grayscale values inside the 
contour for RT-DC measurements
"""
from __future__ import division, print_function, unicode_literals
import numpy as np
import cv2

def get_brightness(images,cont):
    """
    Calculate the mean and std of the grayscale values inside the contour
    
    Parameters
    ----------
    cont: ndarray or list of ndarrays of shape (N,2)
        A 2D array that holds the contour of an event (in pixels)
        e.g. obtained using `mm.contour` where  `mm` is an instance
        of `RTDCBase`. The first and second columns of `cont`
        correspond to the x- and y-coordinates of the contour.
    images: ndarray or list of ndarrays 
        A 2D array that holds the image in form of grayscale values of an event    

    Returns
    -------
    brightn_mean: float or ndarray
        mean brightness in the contour
    brightn_std: float or ndarray
        standard deviation of the grayscale values in the contour
    
    """
    if not type(cont)==list:
        cont = [cont]
    if not type(images)==list:
        images = [images]
    
    msg = 'Nr. of images and contours is not equal'
    assert len(images) == len(cont),msg
    
    #convert contours to OpenCV format
    cont = [a.astype(int).reshape(a.shape[0],1,2) for a in cont] 
    #build a mask / binary image from the contour
    frameMask = np.zeros_like(images[1])
    
    # results are stored in a separate array initialized with nans
    brightn_mean = np.zeros(len(images), dtype=float)*np.nan
    brightn_std = np.zeros(len(images), dtype=float)*np.nan
    for i in range(len(images)):
        frameMask_i = np.copy(frameMask)
        frameMask_i = cv2.drawContours(frameMask_i, cont, i, 255, -1)
        Mean,Std = cv2.meanStdDev(images[i],mask=frameMask_i)
        brightn_mean[i] = Mean
        brightn_std[i] = Std
                   
    if not type(cont)==list:
        # Do not return a list if the input contour was not in a list
        brightn_mean = brightn_mean[0]
        brightn_std = brightn_std[0]
     
    return(brightn,brightn_std)
