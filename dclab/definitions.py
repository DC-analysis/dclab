#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Naming conventions"""
from __future__ import division, print_function, unicode_literals

import copy
import sys

if sys.version_info[0] == 2:
    str_types = (str, unicode)
else:
    str_types = str


class Bool(object):
    """A boolean object"""
    def __init__(self, value):
        if isinstance(value, str_types):
            value = value.lower()
            if value == "false":
                value = False
            elif value == "true":
                value = True
            elif value:
                value = bool(float(value))
            else:
                raise ValueError("empty string")
        else:
            value = bool(float(value))
        self.value = value

    def __repr__(self):
        return "{}".format(self.value)

    def __bool__(self):
        return self.value
    
    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)
    
    def __len__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, Bool):
            return self.value == other.value
        else:
            return self.value == other


class IntList(list):
    """A list of integers"""
    def __init__(self, alist=[]):
        super(IntList, self).__init__()
        if not isinstance(alist, (list, tuple)):
            # we have a string (comma-separated integers)
            alist = alist.strip().strip("[] ").split(",")
        for it in alist:
            if it:
                self.append(it)

    def append(self, value):
        super(IntList, self).append(int(value))


# These are columns that contain a scalar value for each event.
# Other valid columns are "image", "contour", and "trace".
_columns = [
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
   ["fl1_npeaks", "FL-1 number of peaks"],
   ["fl1_pos", "FL-1 position of peak [µs]"],
   ["fl1_width", "FL-1 width [µs]"],
   ["fl2_area", "FL-2 area of peak [a.u.]"],
   ["fl2_dist", "FL-2 distance between two first peaks [µs]"],
   ["fl2_max", "FL-2 maximum [a.u.]"],
   ["fl2_npeaks", "FL-2 number of peaks"],
   ["fl2_pos", "FL-2 position of peak [µs]"],
   ["fl2_width", "FL-2 width [µs]"],
   ["fl3_area", "FL-3 area of peak [a.u.]"],
   ["fl3_dist", "FL-3 distance between two first peaks [µs]"],
   ["fl3_max", "FL-3 maximum [a.u.]"],
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

# Add userdef columns
for _i in range(10):
    _columns.append(["userdef{}".format(_i),
                     "User defined {}".format(_i)
                     ])


column_names = [ _cc[0] for _cc in _columns ]
column_labels = [ _cc[1] for _cc in _columns ]

name2label = {}
for _cc in _columns:
    name2label[_cc[0]] = _cc[1]


# _meta contains all valid configuration keywords for a measurement.
# The variable names depict the corresponding section title in
# a configuration file, e.g.
#
#    [setup]
#    channel width = 20
#    chip region = channel
#    ...
#    [imaging]
#    exposure time = 20
#    ...
#    etc.
_cfg_meta = {
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
        ["no absdiff", Bool, "Avoid OpenCV 'absdiff' for avg-bg-correction"],
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
        ["chip region", str, "Imaged chip region (channel or reservoir)"],
        ["flow rate", float, "Flow rate in channel [µl/s]"],
        ["flow rate sample", float, "Sample flow rate [µl/s]"],
        ["flow rate sheath", float, "Sheath flow rate [µl/s]"],
        ["medium", str, "Medium used (e.g. CellCarrierB, water)"],
        ["module composition", str, "Comma-separated list of modules used"],
        ["software version", str, "Acquisition software with version"],
        ["temperature", float, "Chip temperature [°C]"],
        ["viscosity", float, "Medium viscosity [Pa*s] if 'medium' not given"],
        ],
    }


_cfg_analysis = {
    # filtering parameters
    "filtering": [
        ["hierarchy parent", str, "Hierarchy parent of the data set"],
        ["remove invalid events", Bool, "Remove events with inf/nan values"],
        ["enable filters", Bool, "Enable filtering"],
        ["limit events", Bool, "Upper limit for number of filtered events"],
        ["polygon filters", IntList, "Polygon filter indices"],
        ],
    # Addition user-defined data
    "calculation": [
        ["emodulus model", str, "Model for computing elastic moduli"],
        ["emodulus medium", str, "Medium used (e.g. CellCarrierB, water)"],
        ["emodulus temperature", float, "Chip temperature [°C]"],
        ["emodulus viscosity", float, "Viscosity [Pa*s] if 'medium' unknown"],
        ]
    }


_cfg = copy.deepcopy(_cfg_meta)
_cfg.update(_cfg_analysis)

# dict with section as keys and config parameter names as values
config_keys = {}
for _key in _cfg:
    config_keys[_key] = [ it[0] for it in _cfg[_key] ]

# dict of dicts containing the type of section parameters
config_types = {}
for _key in _cfg:
    config_types[_key] = {}
    for _subkey, _type, __ in _cfg[_key]:
        config_types[_key][_subkey] = _type
