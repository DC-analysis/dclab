"""Check RT-DC datasets for completeness"""
import copy
import functools
import warnings

import h5py
import numpy as np

from .core import RTDCBase
from .fmt_hierarchy import RTDC_Hierarchy
from .load import load_file

from .. import definitions as dfn

#: These sections should be fully present, except for the
#: keys in :data:`OPTIONAL_KEYS`.
DESIRABLE_SECTIONS = {
    "experiment",
    "imaging",
    "online_contour",
    "setup",
}

#: log names that end with these strings are not checked
IGNORED_LOG_NAMES = {
    "_para.ini",
    "_image.ini",
    "FG_Config.mcf",
    "parameters.txt",
    "_SoftwareSettings.ini",
    "dckit-history",
}

#: keys that must be present for every measurement
IMPORTANT_KEYS = {
    "experiment": [
        "date",
        "event count",
        "run index",
        "sample",
        "time"],
    "imaging": [
        "flash device",
        "flash duration",
        "frame rate",
        "pixel size",
        "roi position x",
        "roi position y",
        "roi size x",
        "roi size y"],
    "setup": [
        "channel width",
        "chip region",
        "flow rate",
        "medium"],
}

#: keys that must be present for fluorescence measurements
IMPORTANT_KEYS_FL = {
    "fluorescence": [
        "bit depth",
        "channel count",
        "channels installed",
        "laser count",
        "lasers installed",
        "sample rate",
        "samples per event",
        "signal max",
        "signal min",
        "trace median"],
}

#: maximum line length in log files
LOG_MAX_LINE_LENGTH = 100

#: keys that are optional
OPTIONAL_KEYS = {
    "fluorescence": [
        "baseline 1 offset",
        "baseline 2 offset",
        "baseline 3 offset",
        # name, lambda, power have their own special tests
        "channel 1 name",
        "channel 2 name",
        "channel 3 name",
        "laser 1 lambda",
        "laser 2 lambda",
        "laser 3 lambda",
        "laser 1 power",
        "laser 2 power",
        "laser 3 power",
        ],
    "setup": [
        "temperature",
        "chip identifier",
    ],
    "online_contour": [
        # introduced in 0.34.0
        "bg empty",
    ],
}

#: valid metadata choices
#: .. versionchanged:: 0.29.1
#:    medium not anymore restricted to certain set of choices
VALID_CHOICES = {}


@functools.total_ordering
class ICue(object):
    def __init__(self, msg, level, category, data=None, identifier=None,
                 cfg_section=None, cfg_key=None, cfg_choices=None):
        """Integrity cue"""
        #: human readable message
        self.msg = msg
        #: severity level ("violation", "alert", or "info")
        self.level = level
        #: machine-readable data associated with the check
        self.data = data
        #: fail category
        self.category = category
        #: identifier e.g. for UI manipulation in DCKit
        self.identifier = identifier
        #: section (only for categories "missing metadata", "bad metadata")
        self.cfg_section = cfg_section
        #: key (only for categories "missing metadata", "bad metadata");
        #: can be omitted e.g. to communicate that the entire section is
        #: missing
        self.cfg_key = cfg_key
        #: allowed choices for the specific [section]: key combination
        #: (only for categories "missing metadata", "bad metadata")
        self.cfg_choices = cfg_choices
        if self.cfg_choices is None:
            if (cfg_section in VALID_CHOICES
                    and cfg_key in VALID_CHOICES[cfg_section]):
                self.cfg_choices = VALID_CHOICES[cfg_section][cfg_key]

    def __eq__(self, other):
        leveld = {"info": 0,
                  "violation": 1,
                  "alert": 2,
                  }
        return ((leveld[self.level], self.cfg_section or "",
                 self.cfg_key or "", self.category, self.msg) ==
                (leveld[other.level], other.cfg_section or "",
                 other.cfg_key or "", other.category, other.msg))

    def __lt__(self, other):
        leveld = {"info": 0,
                  "violation": 1,
                  "alert": 2, }
        return ((leveld[self.level], self.cfg_section or "",
                 self.cfg_key or "", self.category, self.msg) <
                (leveld[other.level], other.cfg_section or "",
                 other.cfg_key or "", other.category, other.msg))

    def __repr__(self):
        return "<ICue: '{}' at 0x{}>".format(self.msg, hex(id(self)))

    @staticmethod
    def get_level_summary(cues):
        """For a list of ICue, return the abundance of all levels"""
        levels = {"info": 0,
                  "alert": 0,
                  "violation": 0}
        for cue in cues:
            levels[cue.level] += 1
        return levels


class IntegrityChecker(object):
    def __init__(self, path_or_ds):
        """Check the integrity of a dataset

        The argument must be either a path to an .rtdc or .tdms file
        or an instance of `RTDCBase`. If a path is given, then all
        warnings (e.g. UnknownConfigurationKeyWarning) are catched
        and added to the cue list.

        Usage:

        .. code:: python

            ic = IntegrityChecker("/path/to/data.rtdc")
            cues = ic.check()

        """
        self.warn_cues = []
        if isinstance(path_or_ds, RTDCBase):
            self.ds = path_or_ds
            self.finally_close = False
        else:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                self.ds = load_file(path_or_ds)
                for ww in ws:
                    self.warn_cues.append(ICue(
                        msg="{}: {}".format(ww.category.__name__, ww.message),
                        level="alert",
                        category="warning"))
            self.finally_close = True

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        # close the file
        if self.finally_close and hasattr(self.ds, "__exit__"):
            self.ds.__exit__(type, value, tb)

    @property
    def has_fluorescence(self):
        if ("fluorescence" in self.ds
                or "fl1_max" in self.ds
                or "fl2_max" in self.ds
                or "fl3_max" in self.ds):
            fl = True
        else:
            fl = False
        return fl

    def check(self, **kwargs):
        """Run all checks

        This method calls all class methods that start with `check_`.
        `kwargs` are passed to all methods. Possible options are:

        - "expand_section" `bool`: add a cue for every missing
          metadata key if a section is missing
        """
        cues = []
        funcs = IntegrityChecker.__dict__
        if not len(self.ds) == np.sum(self.ds.filter.all):
            raise NotImplementedError(
                "Integrity checks for datasets with active event filters "
                "are not supported!")
        elif self.ds.__class__ == RTDC_Hierarchy:
            raise NotImplementedError(
                "Integrity checks for 'RTDC_Hierarchy' instances are "
                "not supported!")
        for ff in sorted(funcs.keys()):
            if ff.startswith("check_fl_") and not self.has_fluorescence:
                # skip
                continue
            elif ff.startswith("check_"):
                cues += funcs[ff](self, **kwargs)
        return sorted(self.warn_cues + cues)

    def check_compression(self, **kwargs):
        cues = []
        if self.ds.format == "tdms":
            compression = "None"
            data = {"compressed": 0,
                    "total": 1,
                    "uncompressed": 0,
                    }
        elif self.ds.format == "hdf5":
            def iter_count_compression(h5):
                comp = 0
                noco = 0
                for key in h5:
                    obj = h5[key]
                    if isinstance(obj, h5py.Dataset):
                        if obj.compression is None:
                            noco += 1
                        else:
                            comp += 1
                    elif isinstance(obj, h5py.Group):
                        coi, noi = iter_count_compression(obj)
                        comp += coi
                        noco += noi
                    else:
                        raise ValueError("Unknown object: {}".format(obj))
                return comp, noco

            comp, noco = iter_count_compression(self.ds._h5)
            if noco == 0:
                compression = "All"
            elif comp == 0:
                compression = "None"
            else:
                compression = "Partial ({} of {})".format(comp, noco + comp)
            data = {"compressed": comp,
                    "total": noco + comp,
                    "uncompressed": noco,
                    }
        else:
            compression = "Unknown"
            data = None
        cues.append(ICue(
            msg="Compression: {}".format(compression),
            level="info",
            category="general",
            data=data))
        return cues

    def check_feat_index(self, **kwargs):
        """Up until"""
        cues = []
        lends = len(self.ds)
        if "index" in self.ds:
            if not np.all(self.ds["index"] == np.arange(1, lends + 1)):
                cues.append(ICue(
                    msg="The index feature is not enumerated correctly",
                    level="violation",
                    category="feature data"))
        return cues

    def check_feature_size(self, **kwargs):
        cues = []
        lends = len(self.ds)
        for feat in self.ds.features_innate:
            msg = "Features: wrong event count: '{}' ({} of {})"
            if feat == "trace":
                for tr in list(self.ds["trace"].keys()):
                    if len(self.ds["trace"][tr]) != lends:
                        cues.append(ICue(
                            msg=msg.format("trace/" + tr,
                                           len(self.ds["trace"][tr]),
                                           lends),
                            level="violation",
                            category="feature size"))
            else:
                if len(self.ds[feat]) != lends:
                    cues.append(ICue(
                        msg=msg.format(feat, len(self.ds[feat]), lends),
                        level="violation",
                        category="feature size"))
        return cues

    def check_features_unknown_hdf5(self, **kwargs):
        """check for feature column names"""
        cues = []
        if self.ds.format == "hdf5":
            for feat in self.ds._h5["events"]:
                if not dfn.feature_exists(feat):
                    cues.append(ICue(
                        msg="Features: Unknown key '{}'".format(feat),
                        level="violation",
                        category="feature unknown"))
        return cues

    def check_fl_metadata_channel_names(self, **kwargs):
        cues = []
        for ii in range(1, 4):
            chn = "channel {} name".format(ii)
            fli = "fl{}_max".format(ii)
            if (fli in self.ds
                    and chn not in self.ds.config["fluorescence"]):
                # Channel names must be defined when there is
                # a corresponding fluorescence signal.
                cues.append(ICue(
                    msg="Metadata: Missing key [{}] '{}'".format(
                        "fluorescence", chn),
                    level="alert",
                    category="metadata missing",
                    cfg_section="fluorescence",
                    cfg_key=chn))
            elif (fli not in self.ds
                  and chn in self.ds.config["fluorescence"]):
                # Channel names must not be defined when there is
                # no corresponding fluorescence signal.
                cues.append(ICue(
                    msg="Metadata: Unused key defined [{}] '{}'".format(
                        "fluorescence", chn),
                    level="alert",
                    category="metadata invalid",
                    cfg_section="fluorescence",
                    cfg_key=chn))
        return cues

    def check_fl_num_channels(self, **kwargs):
        cues = []
        # check for number of channels
        if "channel count" in self.ds.config["fluorescence"]:
            chc1 = self.ds.config["fluorescence"]["channel count"]
            chc2 = 0
            for ii in range(1, 4):
                chn = "channel {} name".format(ii)
                ecn = "fl{}_max".format(ii)
                if (chn in self.ds.config["fluorescence"] and
                        ecn in self.ds._events):
                    chc2 += 1
            if chc1 != chc2:
                cues.append(ICue(
                    msg="Metadata: fluorescence channel count inconsistent",
                    level="violation",
                    category="metadata wrong",
                    cfg_section="fluorescence",
                    cfg_key="channel count"))
        return cues

    def check_fl_num_lasers(self, **kwargs):
        cues = []
        # check for number of lasers
        if "laser count" in self.ds.config["fluorescence"]:
            lsc1 = self.ds.config["fluorescence"]["laser count"]
            lsc2 = 0
            for ii in range(1, 4):
                kl = "laser {} lambda".format(ii)
                kp = "laser {} power".format(ii)
                if (kl in self.ds.config["fluorescence"] and
                        kp in self.ds.config["fluorescence"] and
                        self.ds.config["fluorescence"][kp] != 0):
                    lsc2 += 1
            if lsc1 != lsc2:
                cues.append(ICue(
                    msg="Metadata: fluorescence laser count inconsistent",
                    level="violation",
                    category="metadata wrong",
                    cfg_section="fluorescence",
                    cfg_key="laser count"))
        return cues

    def check_fl_samples_per_event(self, **kwargs):
        cues = []
        # check for samples per event
        if "samples per event" in self.ds.config["fluorescence"]:
            spe = self.ds.config["fluorescence"]["samples per event"]
            if "trace" in self.ds:
                for key in self.ds["trace"].keys():
                    spek = self.ds["trace"][key][0].size
                    if spek != spe:
                        cues.append(ICue(
                            msg="Metadata: wrong number of samples per "
                                + "event: {} (expected {}, got {})".format(
                                    key, spe, spek),
                            level="violation",
                            category="metadata wrong",
                            cfg_section="fluorescence",
                            cfg_key="samples per event"))
        return cues

    def check_fl_max_positive(self, **kwargs):
        """Check if all fl?_max values are >0.1"""
        cues = []
        neg_feats = []
        for fl in ['fl1_max', 'fl2_max', 'fl3_max']:
            if fl in self.ds:
                if min(self.ds[fl]) <= 0.1:
                    neg_feats.append(fl)
        if neg_feats:
            cues.append(ICue(
                msg="Negative value for feature(s): {}".format(
                    ", ".join(neg_feats)),
                level="alert",
                category="feature data"))
        return cues

    def check_fl_max_ctc_positive(self, **kwargs):
        """Check if all fl?_max_ctc values are > 0.1"""
        cues = []
        neg_feats = []
        for fl in ['fl1_max_ctc', 'fl2_max_ctc', 'fl3_max_ctc']:
            if fl in self.ds:
                if min(self.ds[fl]) <= 0.1:
                    neg_feats.append(fl)
        if neg_feats:
            cues.append(ICue(
                msg="Negative value for feature(s): {}".format(
                    ", ".join(neg_feats)),
                level="alert",
                category="feature data"))
        return cues

    def check_flow_rate(self, **kwargs):
        """Make sure sheath and sample flow rates add up"""
        cues = []
        if ("setup" in self.ds.config
                and "flow rate" in self.ds.config["setup"]
                and "flow rate sample" in self.ds.config["setup"]
                and "flow rate sheath" in self.ds.config["setup"]):
            frsum = self.ds.config["setup"]["flow rate"]
            frsam = self.ds.config["setup"]["flow rate sample"]
            frshe = self.ds.config["setup"]["flow rate sheath"]
            if not np.allclose(frsum, frsam + frshe):
                for k in ["flow rate", "flow rate sheath", "flow rate sample"]:
                    cues.append(ICue(
                        msg="Metadata: Flow rates don't add up (sheath "
                            + "{:g} + sample {:g} != channel {:g})".format(
                                frshe, frsam, frsum),
                        level="alert",
                        category="metadata wrong",
                        cfg_section="setup",
                        cfg_key=k))
        return cues

    def check_flow_rate_not_zero(self, **kwargs):
        """Make sure the flow rate is not zero"""
        cues = []
        if ("setup" in self.ds.config
                and "flow rate" in self.ds.config["setup"]
                and self.ds.config["setup"]["flow rate"] == 0):
            cues.append(ICue(
                msg="Metadata: Flow rate is zero!",
                level="violation",
                category="metadata wrong",
                cfg_section="setup",
                cfg_key="flow rate"))
        return cues

    def check_fmt_hdf5(self, **kwargs):
        cues = []
        # hdf5-based checks
        if self.ds.format == "hdf5":
            # check meta data of images
            for feat in ["image", "image_bg", "mask"]:
                if feat in self.ds._events:
                    imdat = self.ds[feat]
                    for key, val in [['CLASS', b'IMAGE'],
                                     ['IMAGE_VERSION', b'1.2'],
                                     ['IMAGE_SUBCLASS', b'IMAGE_GRAYSCALE']]:
                        if key not in imdat.attrs:
                            cues.append(ICue(
                                msg="HDF5: '/{}': missing ".format(feat)
                                    + "attribute '{}'".format(key),
                                level="alert",
                                category="format HDF5"))
                        elif imdat.attrs.get_id(key).dtype.char != "S":
                            cues.append(ICue(
                                msg="HDF5: '/{}': attribute '{}' ".format(
                                    feat, key)
                                + "should be fixed-length ASCII string",
                                level="alert",
                                category="format HDF5"))
                        elif imdat.attrs[key] != val:
                            cues.append(ICue(
                                msg="HDF5: '/{}': attribute '{}' ".format(
                                    feat, key)
                                + "should have value '{}'".format(val),
                                level="alert",
                                category="format HDF5"))
            # check length of logs
            with h5py.File(self.ds.path, mode="r") as h5:
                if "logs" in h5:
                    logs = h5["logs"]
                    for logname in logs.keys():
                        # ignore tmds meta data log files
                        lign = [logname.endswith(n) for n in IGNORED_LOG_NAMES]
                        if sum(lign):
                            continue
                        log = logs[logname]
                        for ii in range(len(log)):
                            if len(log[ii]) > LOG_MAX_LINE_LENGTH:
                                cues.append(ICue(
                                    msg="Logs: {} line {} ".format(logname, ii)
                                        + "exceeds maximum line length "
                                        + "{}".format(LOG_MAX_LINE_LENGTH),
                                    level="alert",
                                    category="format HDF5"))
        return cues

    def check_info(self, **kwargs):
        cues = []
        cues.append(ICue(
            msg="Fluorescence: {}".format(self.has_fluorescence),
            level="info",
            category="general"))
        cues.append(ICue(
            msg="Data file format: {}".format(self.ds.format),
            level="info",
            category="general"))
        return cues

    def check_metadata_bad(self, **kwargs):
        cues = []
        # check for ROI size
        if ("imaging" in self.ds.config
                and "roi size x" in self.ds.config["imaging"]
                and "roi size y" in self.ds.config["imaging"]):
            for ii, roi in enumerate(["roi size y", "roi size x"]):
                for feat in ["image", "image_bg", "mask"]:
                    if feat in self.ds:
                        soll = self.ds[feat][0].shape[ii]
                        ist = self.ds.config["imaging"][roi]
                        if soll != ist:
                            cues.append(ICue(
                                msg="Metadata: Mismatch [imaging] "
                                    + "'{}' and feature {} ".format(roi,
                                                                    feat)
                                    + "({} vs {})".format(ist, soll),
                                level="violation",
                                category="metadata wrong",
                                cfg_section="imaging",
                                cfg_key=roi))
        return cues

    def check_metadata_choices(self, **kwargs):
        cues = []
        for sec in VALID_CHOICES:
            for key in VALID_CHOICES[sec]:
                if sec in self.ds.config and key in self.ds.config[sec]:
                    val = self.ds.config[sec][key]
                    if val not in VALID_CHOICES[sec][key]:
                        cues.append(ICue(
                            msg="Metadata: Invalid value [{}] {}: '{}'".format(
                                sec, key, val),
                            level="violation",
                            category="metadata wrong",
                            cfg_section=sec,
                            cfg_key=key,
                            cfg_choices=VALID_CHOICES[sec][key]))
        return cues

    def check_metadata_online_filter_polygon_points_shape(self, **kwargs):
        cues = []
        if "online_filter" in self.ds.config:
            for key in self.ds.config["online_filter"].keys():
                if key.endswith("polygon points"):
                    points = self.ds.config["online_filter"][key]
                    if points.shape[1] != 2 or points.shape[0] < 3:
                        cues.append(ICue(
                            msg="Metadata: Wrong shape [online_filter] "
                                + f"{key}: '{points.shape}'",
                            level="violation",
                            category="metadata wrong",
                            cfg_section="online_filter",
                            cfg_key=key))
        return cues

    def check_metadata_missing(self, expand_section=True, **kwargs):
        cues = []
        # These "must" be present:
        important = copy.deepcopy(IMPORTANT_KEYS)
        if self.has_fluorescence:
            important.update(IMPORTANT_KEYS_FL)
        # A list of sections we would like to investigate
        secs_investiage = list(set(important.keys()) | set(DESIRABLE_SECTIONS))

        for sec in secs_investiage:
            if sec not in self.ds.config and not expand_section:
                cues.append(ICue(
                    msg="Metadata: Missing section '{}'".format(sec),
                    level="violation" if sec in important else "alert",
                    category="metadata missing",
                    cfg_section=sec))
            else:
                for key in dfn.config_keys[sec]:
                    if key not in self.ds.config[sec]:
                        if sec in OPTIONAL_KEYS and key in OPTIONAL_KEYS[sec]:
                            # ignore this key
                            continue
                        elif sec in important and key in important[sec]:
                            level = "violation"
                        else:
                            level = "alert"
                        cues.append(ICue(
                            msg="Metadata: Missing key [{}] '{}'".format(sec,
                                                                         key),
                            level=level,
                            category="metadata missing",
                            cfg_section=sec,
                            cfg_key=key))
        # check for temperature
        if "temp" in self.ds:
            if "temperature" not in self.ds.config["setup"]:
                cues.append(ICue(
                    msg="Metadata: Missing key [setup] 'temperature', "
                        + "because the 'temp' feature is given",
                    level="alert",
                    category="metadata missing",
                    cfg_section="setup",
                    cfg_key="temperature"))
        return cues

    def check_ml_class(self, **kwargs):
        """Try to comput ml_class feature and display error message"""
        cues = []
        if "ml_class" in self.ds:
            try:
                self.ds["ml_class"]
            except ValueError as e:
                cues.append(ICue(
                    msg=e.args[0],
                    level="violation",
                    category="feature data"))
        return cues

    def check_shapein_issue3_bad_medium(self, **kwargs):
        """Some versions of Shape-In stored wrong [setup]: medium

        The problem only affects selection of "CellCarrier" which had
        index 0 and as a result "CellCarrierB" was written to the file.
        This means we only have to check for the Shape-In version and
        whether the medium is 'CellCarrierB'. In DCKit, the user can
        then manually edit the medium.

        Affected Shape-In versions: >=2.2.1.0,<2.2.2.3

        https://github.com/ZELLMECHANIK-DRESDEN/ShapeIn_Issues/issues/3
        """
        cues = []
        medium = self.ds.config["setup"].get("medium", "")
        si_ver = self.ds.config["setup"].get("software version", "")
        si_ver = si_ver.strip("dev")  # for e.g. "2.2.1.0dev"
        if (medium == "CellCarrierB"
                and si_ver in ["2.2.1.0", "2.2.2.0", "2.2.2.1", "2.2.2.2"]):
            cues.append(ICue(
                msg="Metadata: Please verify that 'medium' is really "
                    + "'CellCarrierB' (Shape-In issue #3)",
                level="alert",
                category="metadata wrong",
                identifier="Shape-In issue #3",
                cfg_section="setup",
                cfg_key="medium"))
        return cues


def check_dataset(path_or_ds):
    """Check whether a dataset is complete

    Parameters
    ----------
    path_or_ds: str or pathlib.Path or RTDCBase
        Full path to a dataset on disk or an instance of RTDCBase

    Returns
    -------
    violations: list of str
        Dataset format violations (hard)
    alerts: list of str
        Dataset format alerts (soft)
    info: list of str
        Dataset information
    """
    aler = []
    info = []
    viol = []
    with IntegrityChecker(path_or_ds) as ic:
        # perform all checks
        icues = ic.check(expand_section=False)
        for cue in icues:
            if cue.level == "info":
                info.append(cue.msg)
            elif cue.level == "alert":
                aler.append(cue.msg)
            elif cue.level == "violation":
                viol.append(cue.msg)
    return sorted(viol), sorted(aler), sorted(info)
