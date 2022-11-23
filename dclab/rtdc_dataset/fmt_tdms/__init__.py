"""RT-DC .tdms file format"""
import copy
import pathlib
import time

from ...external.packaging import parse as parse_version

from .exc import ContourIndexingError  # noqa:F401
from .exc import InvalidTDMSFileFormatError  # noqa:F401
from .exc import IncompleteTDMSFileFormatError  # noqa:F401
from .exc import InvalidVideoFileError  # noqa:F401

try:
    import nptdms
except ModuleNotFoundError:
    NPTDMS_AVAILABLE = False
else:
    if parse_version(nptdms.__version__) < parse_version("0.23.0"):
        raise ValueError("Please install nptdms>=0.23.0")
    NPTDMS_AVAILABLE = True
    from .event_contour import ContourColumn
    from .event_image import ImageColumn
    from .event_mask import MaskColumn
    from .event_trace import TraceColumn

import numpy as np

from ... import definitions as dfn
from ...util import hashobj, hashfile

from ..config import Configuration
from ..core import RTDCBase

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

    @staticmethod
    def extract_tdms_config(path,
                            features_available=None,
                            ret_source_files=False,
                            ignore_missing=False):
        """Extract as much metadata as possible for a .tdms dataset

        Parameters
        ----------
        path: str or pathlib.Path
            A path representing the dataset. This could be either a
            .tdms file or an .avi file. The only thing important here
            is the prefix (before the underscore "_") which determines
            the location of the camera.ini and para.ini files.
        features_available: list of str
            List of features known to be available for this dataset.
            Used for defining e.g. fluorescnence metadata.
        ret_source_files: bool
            Return the list of files used to extract metadata from.
        ignore_missing: bool
            Nevermind when para.ini is missing.

        Returns
        -------
        config: .Configuration
            The metadata Configuration instance
        source_paths: str
            List of metadata file paths, only returned when ret_source_files
            is True
        """
        if features_available is None:
            features_available = []
        mid = path.name.split("_")[0]
        # Set up configuration
        config_paths = []
        para_ini = path.with_name(mid + "_para.ini")
        if not para_ini.exists() and not ignore_missing:
            raise IncompleteTDMSFileFormatError(f"Could not find {para_ini}!")
        for pp in [para_ini,
                   path.with_name(mid + "_camera.ini"),
                   path.with_name(mid + "_SoftwareSettings.ini")]:
            if pp.exists():
                config_paths.append(pp)

        tdms_config = Configuration(files=config_paths, disable_checks=True)

        dclab_config = Configuration()

        source_files = copy.copy(config_paths)

        for cfgii in [naming.configmap, naming.config_map_set]:
            for section in cfgii:
                for pname in cfgii[section]:
                    meta = cfgii[section][pname]
                    convfunc = dfn.get_config_value_func(section, pname)
                    if isinstance(meta, tuple):
                        osec, opar = meta
                        if osec in tdms_config and opar in tdms_config[osec]:
                            val = tdms_config[osec].pop(opar)
                            dclab_config[section][pname] = convfunc(val)
                    else:
                        dclab_config[section][pname] = convfunc(meta)

        # Additional information from log file
        rtfdc_log = path.with_name(mid + "_log.ini")
        if rtfdc_log.exists():
            source_files.append(rtfdc_log)
            with rtfdc_log.open("r", errors="replace") as fd:
                loglines = fd.readlines()
            for line in loglines:
                if line.startswith("[EVENT LOG]"):
                    sv = line.split("]")[1].strip()
                    if sv:
                        dclab_config["setup"]["software version"] = sv

        rtfdc_parm = path.with_name("parameters.txt")
        if rtfdc_parm.exists():
            source_files.append(rtfdc_parm)
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
                    sr = int(float(line.split()[1]))
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
                if (fln in features_available
                        and chn not in dclab_config["fluorescence"]):
                    dclab_config["fluorescence"][chn] = "FL{}".format(ii)
            lc = bool(p1) + bool(p2) + bool(p3)
            dclab_config["fluorescence"]["laser count"] = lc
            li = (p1 is not None) + (p2 is not None) + (p3 is not None)
            dclab_config["fluorescence"]["lasers installed"] = li
            dclab_config["fluorescence"]["channels installed"] = 3

        # fluorescence
        if ("fluorescence" in dclab_config
            or "fl1_max" in features_available
            or "fl2_max" in features_available
                or "fl3_max" in features_available):
            # hardware-defined (always the same)
            dclab_config["fluorescence"].setdefault("bit depth", 16)
            dclab_config["fluorescence"].setdefault("laser 1 lambda", 488.)
            dclab_config["fluorescence"].setdefault("laser 2 lambda", 561.)
            dclab_config["fluorescence"].setdefault("laser 3 lambda", 640.)

        # Additional information from commented-out log-file (manual)
        if para_ini.exists():
            text = para_ini.read_text(errors="replace").split("\n")
            lns = [s[1:].strip() for s in text if s.startswith("#")]
            if lns and lns[0] == "[FLUOR]":
                if ("software version" not in dclab_config["setup"]
                        and lns[1].startswith("fRTDC")):
                    dclab_config["setup"]["software version"] = lns[1]
                for ll in lns[2:]:
                    if ("sample rate" not in dclab_config["fluorescence"]
                            and ll.startswith("Samplerate")):
                        val = int(float(ll.split("=")[1]))
                        dclab_config["fluorescence"]["sample rate"] = val
                    elif ("signal min" not in dclab_config["fluorescence"]
                            and ll.startswith("ADCmin")):
                        val = float(ll.split("=")[1])
                        dclab_config["fluorescence"]["signal min"] = val
                    elif ("signal max" not in dclab_config["fluorescence"]
                            and ll.startswith("ADCmax")):
                        val = float(ll.split("=")[1])
                        dclab_config["fluorescence"]["signal max"] = val

        if dclab_config["imaging"].get("frame rate") == 0:
            dclab_config["imaging"].pop("frame rate")

        if dclab_config["setup"].get("flow rate") == 0:
            dclab_config["setup"].pop("flow rate")

        if "channel width" not in dclab_config["setup"]:
            if "channel width" in tdms_config.get("general", {}):
                channel_width = tdms_config["general"]["channel width"]
            elif dclab_config["setup"].get("flow rate", 0) < 0.16:
                channel_width = 20.
            else:
                channel_width = 30.
            dclab_config["setup"]["channel width"] = channel_width

        if "sample" not in dclab_config["experiment"]:
            # Measured sample or user-defined reference
            sample = get_project_name_from_path(path)
            dclab_config["experiment"]["sample"] = sample

        # imaging
        dclab_config["imaging"].setdefault("pixel size", 0.34)

        # medium convention for CellCarrierB
        if ("medium" in dclab_config["setup"] and
                dclab_config["setup"]["medium"].lower() == "cellcarrier b"):
            dclab_config["setup"]["medium"] = "CellCarrierB"

        # replace "+" with ","
        if "module composition" in dclab_config["setup"]:
            mc = dclab_config["setup"]["module composition"]
            if mc.count("+"):
                mc2 = ", ".join([m.strip() for m in mc.split("+")])
                dclab_config["setup"]["module composition"] = mc2

        dclab_config["imaging"].setdefault("flash device", "LED")
        dclab_config["imaging"].setdefault("flash duration", 2.0)
        dclab_config["imaging"].setdefault("roi position x", 0)
        dclab_config["imaging"].setdefault("roi position y", 0)

        if mid.startswith("m") and mid[1] in "0123456789":
            run_index = int(mid.strip("mM"))
        else:
            run_index = 1
        dclab_config["experiment"].setdefault("run index", run_index)

        if ret_source_files:
            return dclab_config, source_files
        else:
            return dclab_config

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

        self.config, config_paths = self.extract_tdms_config(
            self.path,
            features_available=sorted(self._events.keys()),
            ret_source_files=True)
        self._complete_config_with_data()

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
        for pp in sorted(set(log_files)):  # avoid duplicates
            with pp.open("r", errors="replace") as f:
                cfg = [s.strip() for s in f.readlines()]
            self.logs[pp.name] = cfg

    def _complete_config_with_data(self):
        # measurement start time
        tse = self.path.stat().st_mtime
        if "time" in self:
            # correct for duration of experiment
            tse -= self["time"][-1]
        loct = time.localtime(tse)

        # Start time of measurement ('HH:MM:SS')
        timestr = time.strftime("%H:%M:%S", loct)
        self.config["experiment"].setdefault("time", timestr)

        # Date of measurement ('YYYY-MM-DD')
        datestr = time.strftime("%Y-%m-%d", loct)
        self.config["experiment"].setdefault("date", datestr)

        # Number of recorded events
        self.config["experiment"].setdefault("event count", len(self))

        # fmt_tdms
        self.config["fmt_tdms"].setdefault("video frame offset", 1)

        # setup (compatibility to old tdms formats)
        self.config["setup"].setdefault("flow rate", np.nan)

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
    path: str or pathlib.Path
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
