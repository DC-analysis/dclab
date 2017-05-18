#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Computation of volume for RT-DC measurements based on a rotation
of the contours"""
from __future__ import division, print_function, unicode_literals
import numpy as np
import itertools as IT

def get_volume(cont, pos_x, pos_y,pix):
    """Compute  volume according to a matlab script from Geoff Olynyk:
    http://de.mathworks.com/matlabcentral/fileexchange/36525-volrevolve/content/volRevolve.m
    The volume estimation assumes rotational symmetry. 
    Green`s theorem and the Gaussian divergence theorem allow to formulate
    the volume as a line integral.
    
    Parameters
    ----------
    cont: list of ndarrays 
        Each array contains the contour of one event(s) (in pixels)
        e.g. obtained using mm.contour
    pos_x: float or ndarray 
        The x coordinate of the centroid of the event(s)
        e.g. obtained using mm.pos_x
    pos_y: float or ndarray 
        The y coordinate of the centroid of the event(s)
        e.g. obtained using mm.pos_lat
    px_um: float
        The detector pixel size in µm. Set this value to zero
        e.g. obtained using: mm.config["image"]["pix size"]

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
    v_avg = []
    for i in range(len(pos_x)):
        try:
            contour_x = (cont[i])[:,0] #write x coord in a single array
            contour_y = (cont[i])[:,1]#write y coord in a single array
            #Turn the contour around (it has to be conter-clockwise!)
            contour_x = contour_x[::-1]
            contour_y = contour_y[::-1]
            #Move the whole contour down, such that the y coord. of the 
            #centroid becomes 0
            contour_y = contour_y - pos_y[i]/pix
            #Which points are below the x-axis? (y<0)?
            ind_low = np.where(contour_y<0)
            #These points will be shifted up to build an x-axis (wont contribute to volume)
            contour_y_low = np.copy(contour_y)
            contour_y_low[ind_low] = 0
            #Which points are above the x-axis? (y>0)?
            ind_up = np.where(contour_y>0)
            #These points will be shifted up to build an x-axis (wont contribute to volume)
            contour_y_up = np.copy(contour_y)
            contour_y_up[ind_up] = 0
            #Move the contour to the left
            Z = contour_x-pos_x[i]/pix
            #Last point of the contour has to overlap with the first point
            Z = np.hstack([Z,Z[0]])
            Zp = Z[0:-1]
            dZ = Z[1:]-Zp
    
            contour_y_low = np.hstack([contour_y_low,contour_y_low[0]])
            contour_y_up = np.hstack([contour_y_up,contour_y_up[0]])
    
            #Instead of x and y, describe the contour by a Radius vector R and y
            #The Contour will be rotated around the x-axis. Therefore it is
            #Important that the Contour has been shifted onto the x-Axis
            R = np.sqrt(Z**2+contour_y_low**2)
            Rp = R[0:-1]
            dR = R[1:]-Rp
            #4 volume parts
            v1 = dR * dZ * Rp 
            v2 = 2 * dZ * Rp**2
            v3 = (-1) * dR**2 * dZ
            v4 = (-2) * dR * Rp * Zp
            
            V = (np.pi/3) * (v1 + v2 + v3 + v4)
            vol_low = -(np.sum(V))*((pix)**3.0)
            
            R = np.sqrt(Z**2+contour_y_up**2)
            Rp = R[0:-1]
            dR = R[1:]-Rp
            #4 volume parts
            v1 = dR * dZ * Rp 
            v2 = 2 * dZ * Rp**2
            v3 = (-1) * dR**2 * dZ
            v4 = (-2) * dR * Rp * Zp
            
            V = (np.pi/3) * (v1 + v2 + v3 + v4)
            vol_up = -(np.sum(V))*((pix)**3.0)

            v_avg.append((abs(vol_low)+abs(vol_up))/2.0)
        except:
             v_avg.append(np.nan)
    v_avg = np.array(v_avg)
    
    return v_avg

def test_get_volume():
    #Helper definitions to get an ellipse
    def get_ellipse_coords(a=0.0, b=0.0, x=0.0, y=0.0, angle=0.0, k=2):
        """ Draws an ellipse using (360*k + 1) discrete points; based on pseudo code
        given at http://en.wikipedia.org/wiki/Ellipse
        k = 1 means 361 points (degree by degree)
        a = major axis distance,
        b = minor axis distance,
        x = offset along the x-axis
        y = offset along the y-axis
        angle = clockwise rotation [in degrees] of the ellipse;
            * angle=0  : the ellipse is aligned with the positive x-axis
            * angle=30 : rotated 30 degrees clockwise from positive x-axis
        """
        pts = np.zeros((360*k+1, 2))
    
        beta = -angle * np.pi/180.0
        sin_beta = np.sin(beta)
        cos_beta = np.cos(beta)
        alpha = np.radians(np.r_[0.:360.:1j*(360*k+1)])
     
        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)
        
        pts[:, 0] = x + (a * cos_alpha * cos_beta - b * sin_alpha * sin_beta)
        pts[:, 1] = y + (a * cos_alpha * sin_beta + b * sin_alpha * cos_beta)
    
        return pts
    
    def area_of_polygon(x,y):
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    
    def centroid_of_polygon(points):
        """
        http://stackoverflow.com/a/14115494/190597 (mgamba)
        Centroid of polygon: http://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon
        """
        area = area_of_polygon(*zip(*points))
        result_x = 0
        result_y = 0
        N = len(points)
        points = IT.cycle(points)
        x1, y1 = next(points)
        for i in range(N):
            x0, y0 = x1, y1
            x1, y1 = next(points)
            cross = (x0 * y1) - (x1 * y0)
            result_x += (x0 + x1) * cross
            result_y += (y0 + y1) * cross
        result_x /= (area * 6.0)
        result_y /= (area * 6.0)
        return (abs(result_x),  abs(result_y))
    
    major = 10.0
    minor = 5.0
    ellip = get_ellipse_coords(a=major, b=minor, x=minor, y=5.0, angle=0, k=300)
    cx,cy = centroid_of_polygon(ellip) #obtain the centroid (corresponds to 
    #pos_x and pos_lat)
    cx = cx.reshape(1)
    cy = cy.reshape(1)
    
    elliplist = []
    elliplist.append(ellip)
    volume = get_volume(cont=elliplist,pos_x=cx,pos_y=cy,pix=1)
    
    #Analytic solution for volume of ellipsoid:
    V = 4.0/3.0*np.pi*major*minor*minor
    msg = "Calculation of volume is faulty!"
    assert np.allclose(np.array(volume), np.array(V)),msg

    




