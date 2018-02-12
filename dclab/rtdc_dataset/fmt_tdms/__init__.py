#!/usr/bin/python
# -*- coding: utf-8 -*-
"""RT-DC .tdms file format"""
from __future__ import division, print_function, unicode_literals

import io
import os
import time
import sys

import numpy as np
from nptdms import TdmsFile

from ... import definitions as dfn

from ..config import Configuration
from ..core import RTDCBase
from ..util import hashobj, hashfile

from .event_contour import ContourColumn
from .event_image import ImageColumn
from .event_trace import TraceColumn
from . import naming


class RTDC_TDMS(RTDCBase):
    def __init__(self, tdms_path, *args, **kwargs):
        """TDMS file format for RT-DC measurements

        Parameters
        ----------
        tdms_path: str
            Path to a '.tdms' measurement file.
        *args, **kwargs:
            (Keyword) arguments for `RTDCBase`
        """
        # Initialize RTDCBase
        super(RTDC_TDMS, self).__init__(*args, **kwargs)

        # Events is a simple dictionary
        self._events = {}
        self._hash = None
        self.path = tdms_path
        self.title = get_project_name_from_path(tdms_path, append_mx=True)

        # tdms-related convenience properties
        self._fdir = os.path.dirname(self.path)
        self._mid = os.path.basename(self.path).split("_")[0]
        md, mn = os.path.split(self.path)
        self._path_mx = os.path.join(md, mn.split("_")[0])
        
        self._init_data_with_tdms(tdms_path)

        # Add additional features
        # event images
        self._events["image"] = ImageColumn(self)
        # event contours
        self._events["contour"] = ContourColumn(self)
        # event traces
        self._events["trace"] = TraceColumn(self)


    def _init_data_with_tdms(self, tdms_filename):
        """Initializes the current RT-DC data set with a tdms file.
        """
        tdms_file = TdmsFile(tdms_filename)
        # time is always there
        table = "Cell Track"
        datalen = len(tdms_file.object(table, "time").data)
        # Edit naming.dclab2tdms to add features
        for arg in naming.tdms2dclab:
            try:
                data = tdms_file.object(table, arg).data
            except KeyError:
                pass
            else:
                if data is None or len(data)==0:
                    # Fill empty features with zeros. npTDMS treats empty
                    # features in the following way:
                    # - in nptdms 0.8.2, `data` is `None`
                    # - in nptdms 0.9.0, `data` is an array of length 0
                    data = np.zeros(datalen)
                self._events[naming.tdms2dclab[arg]] = data

        # Set up configuration
        tdms_config = Configuration(files=[self._path_mx+"_para.ini",
                                           self._path_mx+"_camera.ini"],
                                    )
        dclab_config = Configuration()
        for section in naming.configmap:
            for pname in naming.configmap[section]:
                meta = naming.configmap[section][pname]
                typ = dfn.config_funcs[section][pname]
                if isinstance(meta, tuple):
                    osec, opar = meta
                    if osec in tdms_config and opar in tdms_config[osec]:
                        val = tdms_config[osec].pop(opar)
                        dclab_config[section][pname] = typ(val)
                else:
                    dclab_config[section][pname] = typ(meta)

        self.config = dclab_config
        self._complete_config_tdms(tdms_config)

        self._init_filters()


    def _complete_config_tdms(self, residual_config={}):
        # experiment
        gmtime = time.gmtime(os.stat(self.path).st_mtime)
        if "date" not in self.config["experiment"]:
            # Date of measurement ('YYYY-MM-DD')
            datestr = time.strftime("%Y-%m-%d", gmtime)
            self.config["experiment"]["date"] = datestr
        if "event count" not in self.config["experiment"]:
            # Number of recorded events
            self.config["experiment"]["event count"] = len(self)
        if "sample" not in self.config["experiment"]:
            # Measured sample or user-defined reference
            sample = get_project_name_from_path(self.path)
            self.config["experiment"]["sample"] = sample
        if "time" not in self.config["experiment"]:
            # Start time of measurement ('HH:MM:SS')
            timestr = time.strftime("%H:%M:%S", gmtime)
            self.config["experiment"]["time"] = timestr
        # fluorescence
        if "fluorescence" in self.config:
            self.config["fluorescence"]["laser 1 lambda"] = 488
            self.config["fluorescence"]["laser 2 lambda"] = 561
            self.config["fluorescence"]["laser 3 lambda"] = 640
        # fmt_tdms
        if "video frame offset" not in self.config["fmt_tdms"]:
            self.config["fmt_tdms"]["video frame offset"] = 1
        # setup (compatibility to old tdms formats)
        if "flow rate" not in self.config["setup"]:
            self.config["setup"]["flow rate"] = np.nan
        if "channel width" not in self.config["setup"]:
            if "channel width" in residual_config["general"]:
                channel_width = residual_config["general"]["channel width"]
            if self.config["setup"]["flow rate"] < 0.16:
                channel_width = 20.
            else:
                channel_width = 30.
            self.config["setup"]["channel width"] = channel_width
        if "temperature" not in self.config["setup"]:
            if "ambient temp. [c]" in residual_config["image"]:
                temp = residual_config["image"]["ambient temp. [c]"]
            elif "ambient temperature [c]" in residual_config["image"]:
                temp = residual_config["image"]["ambient temperature [c]"]
            else:
                temp = np.nan
            self.config["setup"]["temperature"] = temp
        if "viscosity" not in self.config["setup"]:
            self.config["setup"]["viscosity"] = np.nan
        # imaging
        if "pixel size" not in self.config["imaging"]:
            self.config["imaging"]["pixel size"] = 0.34


    @property
    def hash(self):
        """Hash value based on file name and .ini file content"""
        if self._hash is None:
            # Only hash _camera.ini and _para.ini
            fsh = [ self._path_mx+"_camera.ini", self._path_mx+"_para.ini" ]
            tohash = [ hashfile(f) for f in fsh ]
            tohash.append(os.path.basename(self.path))
            # Hash a maximum of ~1MB of the tdms file
            tohash.append(hashfile(self.path, blocksize=65536, count=20))
            self._hash = hashobj(tohash)
        return self._hash


def get_project_name_from_path(path, append_mx=False):
    """Get the project name from a path.
    
    For a path "/home/peter/hans/HLC12398/online/M1_13.tdms" or
    For a path "/home/peter/hans/HLC12398/online/data/M1_13.tdms" or
    without the ".tdms" file, this will return always "HLC12398".
    
    Parameters
    ----------
    path: str
        path to tdms file
    append_mx: bool
        append measurement number, e.g. "M1"
    """
    if path.endswith(".tdms"):
        dirn = os.path.dirname(path)
        mx = os.path.basename(path).split("_")[0]
    elif os.path.isdir(path):
        dirn = path
        mx = ""
    else:
        dirn = os.path.dirname(path)
        mx = ""

    project = ""
    if mx:
        # check para.ini
        para = os.path.join(dirn, mx + "_para.ini")
        if os.path.exists(para):
            with io.open(para) as fd:
                lines = fd.readlines()
            for line in lines:
                if line.startswith("Sample Name ="):
                    project = line.split("=")[1].strip()
                    break
    
    if not project:
        # check if the directory contains data or is online
        root1, trail1 = os.path.split(dirn)
        root2, trail2 = os.path.split(root1)
        _root3, trail3 = os.path.split(root2)
        
        if trail1.lower() in ["online", "offline"]:
            # /home/peter/hans/HLC12398/online/
            project = trail2
        elif ( trail1.lower() == "data" and 
               trail2.lower() in ["online", "offline"] ):
            # this is olis new folder sctructure
            # /home/peter/hans/HLC12398/online/data/
            project = trail3
        else:
            project = trail1

    if append_mx:
        project += " - "+mx
    
    return project


def get_tdms_files(directory):
    """ Recursively find projects based on '.tdms' file endings
    
    Searches the `directory` recursively and return a sorted list
    of all found '.tdms' project files, except fluorescence
    data trace files which end with `_traces.tdms`.
    """
    directory = os.path.realpath(directory)
    tdmslist = list()
    for root, _dirs, files in os.walk(directory):
        for f in files:
            # Exclude traces files of fRT-DC setup
            if (f.endswith(".tdms") and (not f.endswith("_traces.tdms"))):
                tdmslist.append(os.path.realpath(os.path.join(root,f)))
    tdmslist.sort()
    return tdmslist
