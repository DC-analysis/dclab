#!/usr/bin/python
# -*- coding: utf-8 -*-
"""RT-DC .tdms file format"""
from __future__ import division, print_function, unicode_literals

import io
import os
import sys

import numpy as np
from nptdms import TdmsFile

from ..config import Configuration
from ..core import RTDCBase
from ..util import hashobj, hashfile

from .event_contour import ContourColumn
from .event_image import ImageColumn
from .event_trace import TraceColumn
from . import naming


class RTDC_TDMS(RTDCBase):
    def __init__(self, tdms_path):
        """TDMS file format for RT-DC measurements

        Parameters
        ----------
        tdms_path: str
            Path to a '.tdms' measurement file.        
        """
        # Initialize RTDCBase
        super(RTDC_TDMS, self).__init__()

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

        # Add additional columns
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
        # Edit naming.dclab2tdms to add columns
        for arg in naming.tdms2dclab:
            try:
                data = tdms_file.object(table, arg).data
            except KeyError:
                pass
            else:
                if data is None or len(data)==0:
                    # Fill empty columns with zeros. npTDMS treats empty
                    # columns in the following way:
                    # - in nptdms 0.8.2, `data` is `None`
                    # - in nptdms 0.9.0, `data` is an array of length 0
                    data = np.zeros(datalen)
                self._events[naming.tdms2dclab[arg]] = data

        # Set up filtering
        self.config = Configuration(files=[self._path_mx+"_para.ini",
                                           self._path_mx+"_camera.ini"],
                                    rtdc_ds=self)
        self._init_filters()


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
    elif os.path.isdir(path):
        dirn = path
    else:
        dirn = os.path.dirname(path)
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
        mx = os.path.basename(path).split("_")[0]
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
