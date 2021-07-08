"""RT-DC .tdms file format"""
from distutils.version import LooseVersion
import pathlib
import time

from .exc import ContourIndexingError  # noqa:F401
from .exc import InvalidTDMSFileFormatError  # noqa:F401
from .exc import IncompleteTDMSFileFormatError  # noqa:F401
from .exc import InvalidVideoFileError  # noqa:F401

try:
    import nptdms
except ModuleNotFoundError:
    NPTDMS_AVAILABLE = False
else:
    if LooseVersion(nptdms.__version__) < LooseVersion("0.23.0"):
        raise ValueError("Please install nptdms>=0.23.0")
    NPTDMS_AVAILABLE = True
    from .event_contour import ContourColumn
    from .event_image import ImageColumn
    from .event_mask import MaskColumn
    from .event_trace import TraceColumn
    from . import naming

import numpy as np

from ... import definitions as dfn
from ...util import hashobj, hashfile

from ..config import Configuration
from ..core import RTDCBase


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
        if not NPTDMS_AVAILABLE:
            raise ModuleNotFoundError(
                "Package `nptdms` required for TDMS format!")
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
        # event contours (requires image)
        self._events["contour"] = ContourColumn(self)
        # event masks (requires contour)
        self._events["mask"] = MaskColumn(self)
        # event traces
        self._events["trace"] = TraceColumn(self)

    def __contains__(self, key):
        ct = False
        if key in ["contour", "image", "mask", "trace"]:
            # Take into account special cases of the tdms file format:
            # tdms features "image", "trace", "contour", and "mask"
            # evaluate to True (len()!=0) if the data exist on disk
            if key in self._events and self._events[key]:
                ct = True
        else:
            ct = super(RTDC_TDMS, self).__contains__(key)
        return ct

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        del self._events["image"]._image_data

    def _init_data_with_tdms(self, tdms_filename):
        """Initializes the current RT-DC dataset with a tdms file.
        """
        tdms_file = nptdms.TdmsFile(str(tdms_filename))
        # time is always there
        table = "Cell Track"
        # Edit naming.dclab2tdms to add features
        for arg in naming.tdms2dclab:
            try:
                data = tdms_file[table][arg].data
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
        if len(self._events) == 0:
            raise IncompleteTDMSFileFormatError(
                "No usable feature data found in '{}'!".format(tdms_filename))
        # Set up configuration
        config_paths = [self.path.with_name(self._mid + "_para.ini"),
                        self.path.with_name(self._mid + "_camera.ini")]
        for cp in config_paths:
            if not cp.exists():
                raise IncompleteTDMSFileFormatError(
                    "Missing file: {}".format(cp))
        shpin_set = self.path.with_name(self._mid + "_SoftwareSettings.ini")
        if shpin_set.exists():
            config_paths.append(shpin_set)

        tdms_config = Configuration(files=config_paths, disable_checks=True)

        dclab_config = Configuration()

        for cfgii in [naming.configmap, naming.config_map_set]:
            for section in cfgii:
                for pname in cfgii[section]:
                    meta = cfgii[section][pname]
                    typ = dfn.config_funcs[section][pname]
                    if isinstance(meta, tuple):
                        osec, opar = meta
                        if osec in tdms_config and opar in tdms_config[osec]:
                            val = tdms_config[osec].pop(opar)
                            dclab_config[section][pname] = typ(val)
                    else:
                        dclab_config[section][pname] = typ(meta)

        # Additional information from log file
        rtfdc_log = self.path.with_name(self._mid + "_log.ini")
        if rtfdc_log.exists():
            with rtfdc_log.open("r", errors="replace") as fd:
                loglines = fd.readlines()
            for line in loglines:
                if line.startswith("[EVENT LOG]"):
                    sv = line.split("]")[1].strip()
                    if sv:
                        dclab_config["setup"]["software version"] = sv

        rtfdc_parm = self.path.with_name("parameters.txt")
        if rtfdc_parm.exists():
            with rtfdc_parm.open("r", errors="replace") as fd:
                parlines = fd.readlines()
            p1 = None
            p2 = None
            p3 = None
            for line in parlines:
                if line.startswith("pulse_led"):
                    fdur = float(line.split()[1])
                    dclab_config["imaging"]["flash duration"] = fdur
                elif line.startswith("numberofchannels"):
                    nc = int(line.split()[1])
                    dclab_config["fluorescence"]["channel count"] = nc
                elif line.startswith("laser488"):
                    p1 = float(line.split()[1])
                    dclab_config["fluorescence"]["laser 1 lambda"] = 488
                    dclab_config["fluorescence"]["laser 1 power"] = p1
                elif line.startswith("laser561"):
                    p2 = float(line.split()[1])
                    dclab_config["fluorescence"]["laser 2 lambda"] = 561
                    dclab_config["fluorescence"]["laser 2 power"] = p2
                elif line.startswith("laser640"):
                    p3 = float(line.split()[1])
                    dclab_config["fluorescence"]["laser 3 lambda"] = 640
                    dclab_config["fluorescence"]["laser 3 power"] = p3
                elif line.startswith("samplerate"):
                    sr = float(line.split()[1])
                    dclab_config["fluorescence"]["sample rate"] = sr
                elif line.startswith("samplesperframe"):
                    spe = int(line.split()[1])
                    dclab_config["fluorescence"]["samples per event"] = spe
                elif line.startswith("Vmin"):
                    vmin = float(line.split()[1])
                    dclab_config["fluorescence"]["signal min"] = vmin
                elif line.startswith("Vmax"):
                    vmax = float(line.split()[1])
                    dclab_config["fluorescence"]["signal max"] = vmax
                elif line.startswith("median_pmt"):
                    mfs = int(line.split()[1])
                    dclab_config["fluorescence"]["trace median"] = mfs
            # Add generic channel names (independent of lasers)
            for ii in range(1, 4):
                chn = "channel {} name".format(ii)
                fln = "fl{}_max".format(ii)
                if fln in self and chn not in dclab_config["fluorescence"]:
                    dclab_config["fluorescence"][chn] = "FL{}".format(ii)
            lc = bool(p1) + bool(p2) + bool(p3)
            dclab_config["fluorescence"]["laser count"] = lc
            li = (p1 is not None) + (p2 is not None) + (p3 is not None)
            dclab_config["fluorescence"]["lasers installed"] = li
            dclab_config["fluorescence"]["channels installed"] = 3

        # Additional information from commented-out log-file (manual)
        with config_paths[0].open("r", errors="replace") as fd:
            lns = [s[1:].strip() for s in fd.readlines() if s.startswith("#")]
            if lns and lns[0] == "[FLUOR]":
                if ("software version" not in dclab_config["setup"]
                        and lns[1].startswith("fRTDC")):
                    dclab_config["setup"]["software version"] = lns[1]
                for ll in lns[2:]:
                    if ("sample rate" not in dclab_config["fluorescence"]
                            and ll.startswith("Samplerate")):
                        val = float(ll.split("=")[1])
                        dclab_config["fluorescence"]["sample rate"] = val
                    elif ("signal min" not in dclab_config["fluorescence"]
                            and ll.startswith("ADCmin")):
                        val = float(ll.split("=")[1])
                        dclab_config["fluorescence"]["signal min"] = val
                    elif ("signal max" not in dclab_config["fluorescence"]
                            and ll.startswith("ADCmax")):
                        val = float(ll.split("=")[1])
                        dclab_config["fluorescence"]["signal max"] = val

        self.config = dclab_config
        self._complete_config_tdms(tdms_config)

        self._init_filters()

        # Load log files
        log_files = config_paths
        for name in [self._mid + "_events.txt",
                     self._mid + "_log.ini",
                     self._mid + "_SoftwareSettings.ini",
                     "FG_Config.mcf",
                     "parameters.txt"]:
            pl = self.path.with_name(name)
            if pl.exists():
                log_files.append(pl)
        for pp in log_files:
            with pp.open("r", errors="replace") as f:
                cfg = [s.strip() for s in f.readlines()]
            self.logs[pp.name] = cfg

    def _complete_config_tdms(self, residual_config={}):
        # remove zero frame rate
        if ("frame rate" in self.config["imaging"]
                and self.config["imaging"]["frame rate"] == 0):
            self.config["imaging"].pop("frame rate")
        # measurement start time
        tse = self.path.stat().st_mtime
        if "time" in self:
            # correct for duration of experiment
            tse -= self["time"][-1]
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
        if ("fluorescence" in self.config
            or "fl1_max" in self
            or "fl2_max" in self
                or "fl3_max" in self):
            if "bit depth" not in self.config["fluorescence"]:
                # hardware-defined (always the same)
                self.config["fluorescence"]["bit depth"] = 16
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

    @staticmethod
    def can_open(h5path):
        """Check whether a given file is in the .tdms file format"""
        return pathlib.Path(h5path).suffix == ".tdms"

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
            with para.open("r", errors="replace") as fd:
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
