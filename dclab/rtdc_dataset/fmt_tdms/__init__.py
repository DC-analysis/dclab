#!/usr/bin/python
# -*- coding: utf-8 -*-
"""RT-DC .tdms file format"""
from __future__ import division, print_function

import pathlib
import time

from nptdms import TdmsFile
import numpy as np

from ... import definitions as dfn

from ..config import Configuration
from ..core import RTDCBase
from ..util import hashobj, hashfile

from .event_contour import ContourColumn
from .event_image import ImageColumn
from .event_mask import MaskColumn
from .event_trace import TraceColumn
from . import naming


class RTDC_TDMS(RTDCBase):
    def __init__(self, tdms_path, *args, **kwargs):
        """TDMS file format for RT-DC measurements

        Parameters
        ----------
        tdms_path: str or pathlib.Path
            Path to a '.tdms' measurement file.
        *args:
            Arguments for `RTDCBase`
        **kwargs:
            Keyword arguments for `RTDCBase`

        Attributes
        ----------
        path: pathlib.Path
            Path to the experimental dataset (main .tdms file)
        """
        # Initialize RTDCBase
        super(RTDC_TDMS, self).__init__(*args, **kwargs)

        tdms_path = pathlib.Path(tdms_path)
        # Events is a simple dictionary
        self._events = {}
        self._hash = None
        self.path = tdms_path
        self.title = get_project_name_from_path(tdms_path, append_mx=True)

        # tdms-related convenience properties
        self._fdir = tdms_path.parent
        self._mid = tdms_path.name.split("_")[0]

        self._init_data_with_tdms(tdms_path)

        # Add additional features
        # event images
        self._events["image"] = ImageColumn(self)
        # event contours
        self._events["contour"] = ContourColumn(self)
        # event masks (requires contour)
        self._events["mask"] = MaskColumn(self)
        # event traces
        self._events["trace"] = TraceColumn(self)

    def _init_data_with_tdms(self, tdms_filename):
        """Initializes the current RT-DC dataset with a tdms file.
        """
        tdms_file = TdmsFile(str(tdms_filename))
        # time is always there
        table = "Cell Track"
        # Edit naming.dclab2tdms to add features
        for arg in naming.tdms2dclab:
            try:
                data = tdms_file.object(table, arg).data
            except KeyError:
                pass
            else:
                if data is None or len(data) == 0:
                    # Ignore empty features. npTDMS treats empty
                    # features in the following way:
                    # - in nptdms 0.8.2, `data` is `None`
                    # - in nptdms 0.9.0, `data` is an array of length 0
                    continue
                self._events[naming.tdms2dclab[arg]] = data

        # Set up configuration
        tdms_config = Configuration(
            files=[self.path.with_name(self._mid + "_para.ini"),
                   self.path.with_name(self._mid + "_camera.ini")],
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
        tse = self.path.stat().st_mtime - self["time"][-1]
        loct = time.localtime(tse)
        if "date" not in self.config["experiment"]:
            # Date of measurement ('YYYY-MM-DD')
            datestr = time.strftime("%Y-%m-%d", loct)
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
            timestr = time.strftime("%H:%M:%S", loct)
            self.config["experiment"]["time"] = timestr
        # fluorescence
        if "fluorescence" in self.config:
            if "laser 1 power" in self.config["fluorescence"]:
                self.config["fluorescence"]["laser 1 lambda"] = 488.
            if "laser 2 power" in self.config["fluorescence"]:
                self.config["fluorescence"]["laser 2 lambda"] = 561.
            if "laser 3 power" in self.config["fluorescence"]:
                self.config["fluorescence"]["laser 3 lambda"] = 640.
        # fmt_tdms
        if "video frame offset" not in self.config["fmt_tdms"]:
            self.config["fmt_tdms"]["video frame offset"] = 1
        # setup (compatibility to old tdms formats)
        if "flow rate" not in self.config["setup"]:
            self.config["setup"]["flow rate"] = np.nan
        if "channel width" not in self.config["setup"]:
            if "channel width" in residual_config["general"]:
                channel_width = residual_config["general"]["channel width"]
            elif self.config["setup"]["flow rate"] < 0.16:
                channel_width = 20.
            else:
                channel_width = 30.
            self.config["setup"]["channel width"] = channel_width
        # imaging
        if "pixel size" not in self.config["imaging"]:
            self.config["imaging"]["pixel size"] = 0.34
        # medium convention for CellCarrierB
        if ("medium" in self.config["setup"] and
                self.config["setup"]["medium"].lower() == "cellcarrier b"):
            self.config["setup"]["medium"] = "CellCarrierB"
        # replace "+" with ","
        if "module composition" in self.config["setup"]:
            mc = self.config["setup"]["module composition"]
            if mc.count("+"):
                mc2 = ", ".join([m.strip() for m in mc.split("+")])
                self.config["setup"]["module composition"] = mc2

    @property
    def hash(self):
        """Hash value based on file name and .ini file content"""
        if self._hash is None:
            # Only hash _camera.ini and _para.ini
            fsh = [self.path.with_name(self._mid + "_camera.ini"),
                   self.path.with_name(self._mid + "_para.ini")]
            tohash = [hashfile(f) for f in fsh]
            tohash.append(self.path.name)
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
    path = pathlib.Path(path)
    if path.suffix == ".tdms":
        dirn = path.parent
        mx = path.name.split("_")[0]
    elif path.is_dir():
        dirn = path
        mx = ""
    else:
        dirn = path.parent
        mx = ""

    project = ""
    if mx:
        # check para.ini
        para = dirn / (mx + "_para.ini")
        if para.exists():
            with para.open() as fd:
                lines = fd.readlines()
            for line in lines:
                if line.startswith("Sample Name ="):
                    project = line.split("=")[1].strip()
                    break

    if not project:
        # check if the directory contains data or is online
        root1, trail1 = dirn.parent, dirn.name
        root2, trail2 = root1.parent, root1.name
        trail3 = root2.name

        if trail1.lower() in ["online", "offline"]:
            # /home/peter/hans/HLC12398/online/
            project = trail2
        elif (trail1.lower() == "data" and
              trail2.lower() in ["online", "offline"]):
            # this is olis new folder sctructure
            # /home/peter/hans/HLC12398/online/data/
            project = trail3
        else:
            project = trail1

    if append_mx:
        project += " - " + mx

    return project


def get_tdms_files(directory):
    """Recursively find projects based on '.tdms' file endings

    Searches the `directory` recursively and return a sorted list
    of all found '.tdms' project files, except fluorescence
    data trace files which end with `_traces.tdms`.
    """
    path = pathlib.Path(directory).resolve()
    # get all tdms files
    tdmslist = [r for r in path.rglob("*.tdms") if r.is_file()]
    # exclude traces files
    tdmslist = [r for r in tdmslist if not r.name.endswith("_traces.tdms")]
    return sorted(tdmslist)
