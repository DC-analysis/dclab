#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Naming conventions"""
from __future__ import division, print_function, unicode_literals

import copy

from .cfg_funcs import fbool, fintlist, func_types, lcstr


# All configuration keywords editable by the user
CFG_ANALYSIS = {
    # filtering parameters
    "filtering": [
        ["hierarchy parent", str, "Hierarchy parent of the data set"],
        ["remove invalid events", fbool, "Remove events with inf/nan values"],
        ["enable filters", fbool, "Enable filtering"],
        ["limit events", fbool, "Upper limit for number of filtered events"],
        ["polygon filters", fintlist, "Polygon filter indices"],
        ],
    # Addition user-defined data
    "calculation": [
        ["emodulus model", lcstr, "Model for computing elastic moduli"],
        ["emodulus medium", str, "Medium used (e.g. CellCarrierB, water)"],
        ["emodulus temperature", float, "Chip temperature [°C]"],
        ["emodulus viscosity", float, "Viscosity [Pa*s] if 'medium' unknown"],
        ["crosstalk fl21", float, "Fluorescence crosstalk, channel 2 to 1"],
        ["crosstalk fl31", float, "Fluorescence crosstalk, channel 3 to 1"],
        ["crosstalk fl12", float, "Fluorescence crosstalk, channel 1 to 2"],
        ["crosstalk fl32", float, "Fluorescence crosstalk, channel 3 to 1"],
        ["crosstalk fl31", float, "Fluorescence crosstalk, channel 1 to 3"],
        ["crosstalk fl32", float, "Fluorescence crosstalk, channel 2 to 3"],
        ]
    }

# All read-only configuration keywords for a measurement
CFG_METADATA = {
    # All parameters related to the actual experiment
    "experiment": [
        ["date", str, "Date of measurement ('YYYY-MM-DD')"],
        ["event count", int, "Number of recorded events"],
        ["run index", int, "Index of measurement run"],
        ["sample", str, "Measured sample or user-defined reference"],
        ["time", str, "Start time of measurement ('HH:MM:SS')"],
        ],
    # All special keywords related to RT-FDC
    "fluorescence": [
        ["bit depth", int, "Trace bit depth"],
        ["channel count", int, "Number of channels"],
        ["laser 1 power", float, "Laser 1 output power [mW]"],
        ["laser 2 power", float, "Laser 2 output power [mW]"],
        ["laser 3 power", float, "Laser 3 output power [mW]"],
        ["laser 1 lambda", float, "Laser 1 wavelength [nm]"],
        ["laser 2 lambda", float, "Laser 2 wavelength [nm]"],
        ["laser 3 lambda", float, "Laser 3 wavelength [nm]"],
        ["sample rate", float, "Trace sample rate [Hz]"],
        ["signal max", float, "Upper voltage detection limit [V]"],
        ["signal min", float, "Lower voltage detection limit [V]"],
        ["trace median", int, "Rolling median filter size for traces"],
        ],
    # All tdms-related parameters
    "fmt_tdms": [
        ["video frame offset", int, "Missing events at beginning of video"],
        ],
    # All imaging-related keywords
    "imaging": [
        ["exposure time", float, "Sensor exposure time [µs]"],
        ["flash current", float, "Light source current [A]"],
        ["flash device", str, "Light source device type (e.g. green LED)"],
        ["flash duration", float, "Light source flash duration [µs]"],
        ["frame rate", float, "Imaging frame rate [Hz]"],
        ["pixel size", float, "Pixel size [µm]"],
        ["roi position x", float, "Image x coordinate on sensor [px]"],
        ["roi position y", float, "Image y coordinate on sensor [px]"],
        ["roi size x", int, "Image width [px]"],
        ["roi size y", int, "Image height [px]"],
        ],
    # All parameters for online contour extraction from the event images
    "online_contour": [
        ["bin area min", int, "Minium pixel area of binary image event"],
        ["bin kernel", int, "Odd ellipse kernel size, binary image morphing"],
        ["bin margin", int, "Remove margin in x for contour detection"],
        ["bin threshold", int, "Binary threshold for avg-bg-corrected image"],
        ["image blur", int, "Odd sigma for Gaussian blur (21x21 kernel)"],
        ["no absdiff", fbool, "Avoid OpenCV 'absdiff' for avg-bg-correction"],
        ],
    # All online filters
    "online_filter": [
        ["aspect min", float, "Minimum aspect ratio of bounding box"],
        ["aspect max", float, "Maximum aspect ratio of bounding box"],
        ["size_x max", int, "Maximum bounding box size x [µm]"],
        ["size_y max", int, "Maximum bounding box size y [µm]"],
        ["size_x min", int, "Minimum bounding box size x [µm]"],
        ["size_y min", int, "Minimum bounding box size y [µm]"],
        ],
    # All setup-related keywords, except imaging
    "setup": [
        ["channel width", float, "Width of microfluidic channel [µm]"],
        ["chip region", lcstr, "Imaged chip region (channel or reservoir)"],
        ["flow rate", float, "Flow rate in channel [µL/s]"],
        ["flow rate sample", float, "Sample flow rate [µL/s]"],
        ["flow rate sheath", float, "Sheath flow rate [µL/s]"],
        ["medium", str, "Medium used (e.g. CellCarrierB, water)"],
        ["module composition", str, "Comma-separated list of modules used"],
        ["software version", str, "Acquisition software with version"],
        ["temperature", float, "Chip temperature [°C]"],
        ["viscosity", float, "Medium viscosity [Pa*s] if 'medium' not given"],
        ],
    }

# List of scalar features (one number per event).
# The non-scalar features "image", "contour", and "trace" are not listed here.
FEATURES = [
   ["area_cvx", "Convex area [px]"],
   ["area_msd", "Measured area [px]"],
   ["area_ratio", "Convex to measured area ratio"],
   # area_um is the convex area per definition
   ["area_um", "Area [µm²]"],
   ["aspect", "Aspect ratio of bounding box"],
   ["bright_avg", "Brightness average within contour [a.u.]"],
   ["bright_sd", "Brightness SD  within contour [a.u.]"],
   ["circ", "Circularity"],
   ["deform", "Deformation"],
   ["emodulus", "Young's Modulus [kPa]"],
   # fl*_area, fl*_pos, and fl*_width values correspond to the
   # object for which the contour was found. For high concentrations,
   # these values could be error-prone due to the assignment from
   # false objects. 
   ["fl1_area", "FL-1 area of peak [a.u.]"],
   # fl1_dist is set to zero if there is only one peak
   ["fl1_dist", "FL-1 distance between two first peaks [µs]"],
   ["fl1_max", "FL-1 maximum [a.u.]"],
   ["fl1_max_ctc", "FL-1 maximum, crosstalk-corrected [a.u.]"],
   ["fl1_npeaks", "FL-1 number of peaks"],
   ["fl1_pos", "FL-1 position of peak [µs]"],
   ["fl1_width", "FL-1 width [µs]"],
   ["fl2_area", "FL-2 area of peak [a.u.]"],
   ["fl2_dist", "FL-2 distance between two first peaks [µs]"],
   ["fl2_max", "FL-2 maximum [a.u.]"],
   ["fl2_max_ctc", "FL-2 maximum, crosstalk-corrected [a.u.]"],
   ["fl2_npeaks", "FL-2 number of peaks"],
   ["fl2_pos", "FL-2 position of peak [µs]"],
   ["fl2_width", "FL-2 width [µs]"],
   ["fl3_area", "FL-3 area of peak [a.u.]"],
   ["fl3_dist", "FL-3 distance between two first peaks [µs]"],
   ["fl3_max", "FL-3 maximum [a.u.]"],
   ["fl3_max_ctc", "FL-3 maximum, crosstalk-corrected [a.u.]"],
   ["fl3_npeaks", "FL-3 number of peaks"],
   ["fl3_pos", "FL-3 position of peak [µs]"],
   ["fl3_width", "FL-3 width [µs]"],
   ["frame", "Video frame number"],
   # The inertia ratios of the event contours are defined by the
   # central second order moments of area. 
   ["inert_ratio_cvx", "Inertia ratio of convex contour sqrt(m20/m02)"],
   ["inert_ratio_raw", "Inertia ratio of raw contour sqrt(m20/m02)"],
   ["index", "Event index"],
   ["ncells", "Number of cells in image"],
   ["pc1", "Principal component 1"],
   ["pc2", "Principal component 2"],
   ["pos_x", "Position along channel axis [µm]"],
   ["pos_y", "Position lateral in channel [µm]"],
   ["size_x", "Bounding box size x [µm]"],
   ["size_y", "Bounding box size y [µm]"],
   ["time", "Event time [s]"],
   ["volume", "Volume [µm³]"],
   ]
# Add userdef features
for _i in range(10):
    FEATURES.append(["userdef{}".format(_i),
                     "User defined {}".format(_i)
                     ])

# List of fluorescence traces
FLUOR_TRACES = [
    "fl1_median",
    "fl1_raw",
    "fl2_median",
    "fl2_raw",    
    "fl3_median",
    "fl3_raw",
    ]


# CFG convenience lists and dicts
_cfg = copy.deepcopy(CFG_METADATA)
_cfg.update(CFG_ANALYSIS)
# dict with section as keys and config parameter names as values
config_keys = {}
for _key in _cfg:
    config_keys[_key] = [ it[0] for it in _cfg[_key] ]
# dict of dicts containing functions to convert input data
config_funcs = {}
for _key in _cfg:
    config_funcs[_key] = {}
    for _subkey, _type, __ in _cfg[_key]:
        config_funcs[_key][_subkey] = _type
# dict of dicts containing the type of section parameters
config_types = {}
for _key in _cfg:
    config_types[_key] = {}
    for _subkey, _type, __ in _cfg[_key]:
        if _type in func_types:
            _type = func_types[_type]
        config_types[_key][_subkey] = _type

# FEATURE convenience lists and dicts
feature_names = [ _cc[0] for _cc in FEATURES ]
feature_labels = [ _cc[1] for _cc in FEATURES ]
feature_name2label = {}
for _cc in FEATURES:
    feature_name2label[_cc[0]] = _cc[1]
