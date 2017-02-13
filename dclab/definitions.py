#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This file contains basic definitions and naming conventions.
"""
from __future__ import division, print_function, unicode_literals

import numpy as np


def GetKnownIdentifiers():
    return uid


### Define Standard name maps
# Unique identifier (UID)
uid = [
        "AreaPix",
        "Area",
        "Area Ratio",
        "Aspect",
        "Circ",
        "Defo",
        "Frame",
        "Pos Lat",
        "Pos x",
        "Time",
        "FC0max",
        "FC0width",
        "FL-1max",
        "FL-1width",
        "FL-2max",
        "FL-2width",
        "FL-3max",
        "FL-3width",
        "FL-1area",
        "FL-2area",
        "FL-3area",
        "FL-1pos",
        "FL-2pos",
        "FL-3pos",
        "FL-1npeaks",
        "FL-2npeaks",
        "FL-3npeaks",
        "FL-1dpeaks",
        "FL-2dpeaks",
        "FL-3dpeaks",
        "NrOfCells",
        "x-size",
        "y-size",
        "Brightness",
        "BrightnessSD",
        "Inertia Ratio",
        "Inertia Ratio Raw",
        "UserDef1",
        "UserDef2",
        "UserDef3",
        "UserDef4",
        "UserDef5",
        "UserDef6",
        "UserDef7",
        "UserDef8",
        "UserDef9",
        "UserDef0",
        "PC1",
        "PC2",
        "Index",
        ]
# Axes label (same order as UID)
axl = [
        u"Area [px²]",
        u"Area [µm²]",
        u"Convex to measured area ratio",
        u"Aspect ratio of bounding box",
        u"Circularity",
        u"Deformation",
        u"Frame number",
        u"Lateral position in channel [µm]",
        u"Position along channel axis [µm]",
        u"Frame time [s]",
        u"Fluorescence intensity maximum [a.u.]",
        u"Fluorescence peak width [µs]",
        u"FL-1 maximum [a.u.]",
        u"FL-1 width [µs]",
        u"FL-2 maximum [a.u.]",
        u"FL-2 width [µs]",
        u"FL-3 maximum [a.u.]",
        u"FL-3 width [µs]",
        u"FL-1 area of peak [a.u.]",
        u"FL-2 area of peak [a.u.]",
        u"FL-3 area of peak [a.u.]",
        u"FL-1 position of peak [µs]",
        u"FL-2 position of peak [µs]",
        u"FL-3 position of peak [µs]",
        u"FL-1 number of peaks",
        u"FL-2 number of peaks",
        u"FL-3 number of peaks",
        u"FL-1 distance between two first peaks [µs]",
        u"FL-2 distance between two first peaks [µs]",
        u"FL-3 distance between two first peaks [µs]",
        u"Number of cells in image",
        u"Bounding box x-size [µm]",
        u"Bounding box y-size [µm]",
        u"Brightness in contour [a.u.]",
        u"Brightness SD [a.u.]",
        u"Inertia ratio sqrt(m20/m02)",
        u"Raw inertia ratio sqrt(m20/m02)",
        u"User defined 1",
        u"User defined 2",
        u"User defined 3",
        u"User defined 4",
        u"User defined 5",
        u"User defined 6",
        u"User defined 7",
        u"User defined 8",
        u"User defined 9",
        u"User defined 0",
        u"Principal component 1",
        u"Principal component 2",
        u"Event index"
       ]
# Unique RTDC_DataSet variable names (same order as UID)
rdv = [
        "area",
        "area_um",
        "area_ratio",
        "aspect",
        "circ",
        "deform",
        "frame",
        "pos_lat",
        "pos_x",
        "time",
        "fc0m",
        "fc0w",
        "fl1m",
        "fl1w",
        "fl2m",
        "fl2w",
        "fl3m",
        "fl3w",
        "fl1a",
        "fl2a",
        "fl3a",
        "fl1p",
        "fl2p",
        "fl3p",
        "fl1n",
        "fl2n",
        "fl3n",
        "fl1d",
        "fl2d",
        "fl3d",
        "ncells",
        "size_x",
        "size_y",
        "br",
        "brSD",
        "inRatio",
        "inRatioRaw",
        "userDef1",
        "userDef2",
        "userDef3",
        "userDef4",
        "userDef5",
        "userDef6",
        "userDef7",
        "userDef8",
        "userDef9",
        "userDef0",
        "pc1",
        "pc2",
        "index"
       ]
# tdms file definitions (same order as UID)
# group, [names], lambda
# The order of [names] must be the same as the order of the arguments
# for lambda!
tfd = [
        # area -> area in pixels
        ["Cell Track",
         "area",
         lambda x: x
        ],
        # area_um (set by RTDC_DataSet)
        ["Cell Track",
         "area",
         lambda x: np.zeros(x.shape) # set to zero
         ],
        # area_ratio
        ["Cell Track",
         ["area", "raw area"],
         lambda area, area_raw: area/area_raw
         ],
        # aspect
        ["Cell Track",
         ["ax1", "ax2"], #(perpendicular to flow, parallel to flow)
         lambda ax1, ax2: ax2 / ax1
         ],
        # circ
        ["Cell Track",
         "circularity",
         lambda x: x
         ],
        # deform
        ["Cell Track",
         "circularity",
         lambda x: 1-x
         ],
        # frame (the time column is actually the image frame number)
        ["Cell Track",
         "time",
         lambda x: x
        ],
        # pos_lat
        ["Cell Track",
         "y",
         lambda x: x
         ],
        # pos_x
        ["Cell Track",
         "x",
         lambda x: x
         ],
        # time: set to zero, is computed later using config key "Frame Rate"
        ["Cell Track",
         "time",
         lambda x: np.zeros(x.shape) # sic
         ],
        # FC0 maxiumum channel
        ["Cell Track",
         "FC0_max",
         lambda x: x
        ],
        # FC0 width channel
        ["Cell Track",
         "FC0_width",
         lambda x: x
        ],
        # For 3-channel setup use FL-1 .. FL-3 annotation
        # FL-1 maximum of peak (green channel)
        ["Cell Track",
         "FL1max",
         lambda x: x
        ],
        # FL-1 width channel
        ["Cell Track",
         "FL1width",
         lambda x: x
        ],
        # FL-2 maximum of peak (orange channel)
        ["Cell Track",
         "FL2max",
         lambda x: x
        ],
        # FL-2 width channel
        ["Cell Track",
         "FL2width",
         lambda x: x
        ],
        # FL-3 maximum of peak (red channel)
        ["Cell Track",
         "FL3max",
         lambda x: x
        ],
        # FL-3 width channel
        ["Cell Track",
         "FL3width",
         lambda x: x
        ],
        
        # FL-1 area channel
        ["Cell Track",
         "FL1area",
         lambda x: x
        ],
        
        # FL-2 area channel
        ["Cell Track",
         "FL2area",
         lambda x: x
        ],
        
        # FL-3 area channel
        ["Cell Track",
         "FL3area",
         lambda x: x
        ],
        
        # FL-1 position channel
        ["Cell Track",
         "FL1pos",
         lambda x: x
        ],
        
        # FL-2 position channel
        ["Cell Track",
         "FL2pos",
         lambda x: x
        ],
        
        # FL-3 position channel
        ["Cell Track",
         "FL3pos",
         lambda x: x
        ],
        
        # FL-1 number of peaks
        ["Cell Track",
         "FL1npeaks",
         lambda x: x
        ],
        
        # FL-2 number of peaks
        ["Cell Track",
         "FL2npeaks",
         lambda x: x
        ],
        
        # FL-3 number of peaks
        ["Cell Track",
         "FL3npeaks",
         lambda x: x
        ],
        
        # FL-1 distance two first peaks
        ["Cell Track",
         "FL1dpeaks",
         lambda x: x
        ],
        
        # FL-2 distance two first peaks
        ["Cell Track",
         "FL2dpeaks",
         lambda x: x
        ],
        
        # FL-3 distance two first peaks
        ["Cell Track",
         "FL3dpeaks",
         lambda x: x
        ],
        
        # Number of cells in image
        ["Cell Track",
         "NrOfCells",
         lambda x: x
        ],
        # Bounding box x-size
        ["Cell Track",
         "ax2", # parallel to flow
         lambda x:x
         ],
        # Bounding box y-size
        ["Cell Track",
         "ax1", # perpendicular to flow
         lambda x:x
         ],
        # Brightness value determined by fRT-DC setup
        ["Cell Track",
         "Brightness",
         lambda x:x
        ],
        # Brightness SD value determined by fRT-DC setup
        ["Cell Track",
         "BrightnessSD",
         lambda x:x
        ],
        # Inertia Ratio sqrt(mu20 / mu02)
        ["Cell Track",
         "InertiaRatio",
         lambda x:x
        ],
        # Inertia Ratio sqrt(mu20 / mu02)
        ["Cell Track",
         "InertiaRatioRaw",
         lambda x:x
        ],
        # User Defined Values
        # e.g. Compensated Data
        ["Cell Track",
         "UserDef1",
         lambda x:x
        ],
        # -
        ["Cell Track",
         "UserDef2",
         lambda x:x
        ],
        ["Cell Track",
         "UserDef3",
         lambda x:x
        ],
        # -
        ["Cell Track",
         "UserDef4",
         lambda x:x
        ],
        ["Cell Track",
         "UserDef5",
         lambda x:x
        ],
        # -
        ["Cell Track",
         "UserDef6",
         lambda x:x
        ],
        ["Cell Track",
         "UserDef7",
         lambda x:x
        ],
        # -
        ["Cell Track",
         "UserDef8",
         lambda x:x
        ],
        ["Cell Track",
         "UserDef9",
         lambda x:x
        ],
        # -
        ["Cell Track",
         "UserDef0",
         lambda x:x
        ],
        ["Cell Track",
         "PC1",
         lambda x:x
        ],
        # -
        ["Cell Track",
         "PC2",
         lambda x:x
        ],
        # Add event index, use length of frame index
        ["Cell Track",
         "time",
         lambda x: np.arange(1,x.shape[0]+1)
         ],
        ]

# traces_tdms file definitions
# The second column should not contain duplicates! - even if the 
# entries in the first columns are different.
tr_data = [["fluorescence traces", "FL1raw"],
           ["fluorescence traces", "FL2raw"],
           ["fluorescence traces", "FL3raw"],
           ["fluorescence traces", "FL1med"],
           ["fluorescence traces", "FL2med"],
           ["fluorescence traces", "FL3med"],
        ]

# mapping `Measuement` class attributes to configuration file names
cfgmap = dict()        # area_um -> Area
cfgmaprev = dict()     # Area -> area_um
axlabels = dict()      # Area -> Cell Size [µm²]
axlabelsrev = dict()   # Cell Size [µm²] -> Area

# here the name maps are defined
for _u,_a,_r in zip(uid, axl, rdv):
    cfgmap[_r] = _u
    cfgmaprev[_u] = _r
    axlabels[_u] = _a
    axlabelsrev[_a] = _u
