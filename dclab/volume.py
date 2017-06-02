#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Computation of volume for RT-DC measurements based on a rotation
of the contours"""
from __future__ import division, print_function, unicode_literals
import numpy as np

def get_volume(cont, pos_x, pos_y,pix):
    """Compute  volume according to a matlab script from Geoff Olynyk:
    http://de.mathworks.com/matlabcentral/fileexchange/36525-volrevolve/content/volRevolve.m
    The volume estimation assumes rotational symmetry. 
    Green`s theorem and the Gaussian divergence theorem allow to formulate
    the volume as a line integral.
    
    Parameters
    ----------
    cont: list of ndarrays 
        Each array in the list contains the contours of one event (in pixels)
        e.g. obtained using mm.contour (mm is an instance of RTDC_DataSet)
    pos_x: float or ndarray 
        The x coordinate(s) of the centroid of the event(s)
        e.g. obtained using mm.pos_x  (mm is an instance of RTDC_DataSet)
    pos_y: float or ndarray 
        The y coordinate(s) of the centroid of the event(s)
        e.g. obtained using mm.pos_lat  (mm is an instance of RTDC_DataSet)
    px_um: float
        The detector pixel size in µm.
        e.g. obtained using: mm.config["image"]["pix size"]  (mm is an instance 
        of RTDC_DataSet)

    Returns
    -------
    volume: float or ndarray
        volume in um^3
    
    Notes
    -----
    The computation of the volume is based on a full rotation of the upper half 
    of the contour to obtain volume. In the same manner the lower part of the
    contour is rotated. Both volumes are then averaged
    
    """
    #Convert float input to arrays:
    pos_x = np.atleast_1d(pos_x)
    pos_y = np.atleast_1d(pos_y)
    v_avg = np.zeros_like(pos_x,dtype=float)*np.nan

    def vol_helper(contour_y,Z,Zp,dZ,pix):  
        #Instead of x and y, describe the contour by a Radius vector R and y
        #The Contour will be rotated around the x-axis. Therefore it is
        #Important that the Contour has been shifted onto the x-Axis             
        R = np.sqrt(Z**2+contour_y**2)
        Rp = R[0:-1]
        dR = R[1:]-Rp
        #4 volume parts
        v1 = dR * dZ * Rp 
        v2 = 2 * dZ * Rp**2
        v3 = (-1) * dR**2 * dZ
        v4 = (-2) * dR * Rp * Zp
        
        V = (np.pi/3) * (v1 + v2 + v3 + v4)
        vol = -(np.sum(V))*((pix)**3.0)
        return vol

    for i in range(len(pos_x)):
        if len((cont[i])[:,0])>=4:
            contour_x = (cont[i])[:,0] #write x coord in a single array
            contour_y = (cont[i])[:,1]#write y coord in a single array
            #Turn the contour around (it has to be counter-clockwise!)
            contour_x = contour_x[::-1]
            contour_y = contour_y[::-1]
            #Move the whole contour down, such that the y coord. of the 
            #centroid becomes 0
            contour_y = contour_y - pos_y[i]/pix
            #Which points are below the x-axis? (y<0)?
            ind_low = np.where(contour_y<0)
            #These points will be shifted up to y=0 to build an x-axis (wont contribute to volume)
            contour_y_low = np.copy(contour_y)
            contour_y_low[ind_low] = 0
            #Which points are above the x-axis? (y>0)?
            ind_up = np.where(contour_y>0)
            #These points will be shifted down to y=0 to build an x-axis (wont contribute to volume)
            contour_y_up = np.copy(contour_y)
            contour_y_up[ind_up] = 0
            #Move the contour to the left
            Z = contour_x-pos_x[i]/pix
            #Last point of the contour has to overlap with the first point
            Z = np.hstack([Z,Z[0]])
            Zp = Z[0:-1]
            dZ = Z[1:]-Zp
                  
            #Last point of the contour has to overlap with the first point    
            contour_y_low = np.hstack([contour_y_low,contour_y_low[0]])
            contour_y_up = np.hstack([contour_y_up,contour_y_up[0]])
    
            vol_low = vol_helper(contour_y_low,Z,Zp,dZ,pix)
            vol_up = vol_helper(contour_y_up,Z,Zp,dZ,pix)
                            
            v_avg[i] = ((abs(vol_low)+abs(vol_up))/2.0)
        else: #If the contour has less than 4 pixels, the computation will fail
             v_avg[i] = (np.nan)
                
    return v_avg
