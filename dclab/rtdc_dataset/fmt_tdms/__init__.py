#!/usr/bin/python
# -*- coding: utf-8 -*-
"""RT-DC .tdms file format"""
from __future__ import division, print_function, unicode_literals

import hashlib
import os
import sys

import numpy as np
from nptdms import TdmsFile

from dclab import definitions as dfn
from ..config import Configuration
from ..core import RTDCBase, obj2str, hashfile

from .event_contour import ContourColumn
from .event_image import ImageColumn
from .event_trace import TraceColumn



class RTDC_TDMS(RTDCBase):
    def __init__(self, tdms_path):
        """
        Parameters
        ----------
        tdms_path: str
            Path to a '.tdms' file. Only one of `tdms_path and `ddict` can
            be specified.
        ddict: dict
            Dictionary with keys from `dclab.definitions.uid` (e.g. "area", "defo")
            with which the class will be instantiated. Not '.tdms' file is required.
            The configuration is set to the default configuration fo dclab.
        
        Notes
        -----
        Besides the filter arrays for each data column, there is a manual
        boolean filter array ``RTDCBase._filter_manual`` that can be edited
        by the user - a boolean value of ``False`` means that the event is 
        excluded from all computations.
        
        """
        super(RTDC_TDMS, self).__init__()
        self._events = {}

        # Initialize variables and generate hashes
        self.tdms_filename = tdms_path
        self.path = tdms_path
        self.filename = tdms_path
        self.name = os.path.split(tdms_path)[1].split(".tdms")[0]
        self.fdir = os.path.dirname(tdms_path)
        mx = os.path.join(self.fdir, self.name.split("_")[0])
        self.title = u"{} - {}".format(get_project_name_from_path(tdms_path),
                                       os.path.split(mx)[1])
        fsh = [ tdms_path, mx+"_camera.ini", mx+"_para.ini" ]
        self.file_hashes = [(f, hashfile(f)) for f in fsh if os.path.exists(f)]
        ihasher = hashlib.md5()
        ihasher.update(obj2str(tdms_path))
        ihasher.update(obj2str(self.file_hashes))
        self.identifier = ihasher.hexdigest()

        self._init_data_with_tdms(tdms_path)

        # event images
        self._events["image"] = ImageColumn(self)
        # event contours
        self._events["contour"] = ContourColumn(self)
        # event traces
        self._events["trace"] = TraceColumn(self)


    def _init_data_with_tdms(self, tdms_filename):
        """ Initializes the current RT-DC data set with a tdms file.
        """
        tdms_file = TdmsFile(tdms_filename)
        ## Set all necessary internal parameters as defined in
        ## definitions.py
        ## Note that this is meta-programming. If you want to add a
        ## different column from tdms files, then edit definitions.py:
        ## -> uid, axl, rdv, tfd
        # time is always there
        datalen = len(tdms_file.object("Cell Track", "time").data)
        for ii, group in enumerate(dfn.tfd):
            # ii iterates through the data that we could possibly extract
            # from a the tdms file.
            # The `group` contains all information necessary to extract
            # the data: table name, used column names, method to compute
            # the desired data from the columns.
            table = group[0]
            if not isinstance(group[1], list):
                # just for standards
                group[1] = [group[1]]
            func = group[2]
            args = []
            try:
                for arg in group[1]:
                    data = tdms_file.object(table, arg).data
                    if data is None or len(data)==0:
                        # Fill empty columns with zeros. npTDMS treats empty
                        # columns in the following way:
                        # - in nptdms 0.8.2, `data` is `None`
                        # - in nptdms 0.9.0, `data` is an array of length 0
                        data = np.zeros(datalen)
                    args.append(data)
            except KeyError:
                # set it to zero
                func = lambda x: x
                args = [np.zeros(datalen)]
            finally:
                self._events[dfn.rdv[ii]] = func(*args)

        # Set up filtering
        mx = os.path.join(self.fdir, self.name.split("_")[0])
        self.config = Configuration(files=[mx+"_para.ini", mx+"_camera.ini"],
                                    rtdc_ds=self)

        self._init_filters()


def get_project_name_from_path(path):
    """Get the project name from a path.
    
    For a path "/home/peter/hans/HLC12398/online/M1_13.tdms" or
    For a path "/home/peter/hans/HLC12398/online/data/M1_13.tdms" or
    without the ".tdms" file, this will return always "HLC12398".
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
