#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Check RT-DC datasets for completeness"""
from __future__ import unicode_literals

import h5py

from .core import RTDCBase
from .load import load_file

from .. import definitions as dfn


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


#: log names that end with these strings are not checked
IGNORED_LOG_NAMES = {
    "_para.ini",
    "_image.ini",
    "FG_Config.mcf",
    "parameters.txt",
    "_SoftwareSettings.ini"
    }


def check_dataset(path_or_ds):
    """Check whether a dataset is complete

    Parameters
    ----------
    path_or_ds: str or RTDCBase
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
    if isinstance(path_or_ds, RTDCBase):
        ds = path_or_ds
    else:
        ds = load_file(path_or_ds)
    # check for meta data types
    for sec in ds.config:
        for key in ds.config[sec]:
            if sec in dfn.CFG_ANALYSIS:
                # TODO:
                # - properly test against analysis keywords
                #   (filtering, calculation)
                pass
            elif (sec not in dfn.config_keys or
                  key not in dfn.config_keys[sec]):
                viol.append("Metadata: Unknown key [{}] '{}'".format(sec, key))
            elif not isinstance(ds.config[sec][key],
                                dfn.config_types[sec][key]):
                viol.append("Metadata: Datatype of [{}] '{}'".format(sec, key)
                            + "must be '{}'".format(dfn.config_types[sec][key])
                            )
    # check existence of meta data keys
    # These "must" be present:
    tocheck = IMPORTANT_KEYS
    # These sections "should" be fully present
    tocheck_sec_aler = ["experiment", "imaging", "online_contour", "setup"]
    # should we also check for fluorescence keys?
    if ("fluorescence" in ds.config or
        "fl1_max" in ds._events or
        "fl2_max" in ds._events or
            "fl3_max" in ds._events):
        info.append("Fluorescence: True")
        tocheck = tocheck.copy()
        tocheck.update(IMPORTANT_KEYS_FL)
        # check for number of channels
        if "channel count" in ds.config["fluorescence"]:
            chc1 = ds.config["fluorescence"]["channel count"]
            chc2 = 0
            for ii in range(1, 4):
                chn = "channel {} name".format(ii)
                ecn = "fl{}_max".format(ii)
                if (chn in ds.config["fluorescence"] and
                        ecn in ds._events):
                    chc2 += 1
            if chc1 != chc2:
                msg = "Metadata: fluorescence channel count inconsistent"
                viol.append(msg)
        # check for number of lasers
        if "laser count" in ds.config["fluorescence"]:
            lsc1 = ds.config["fluorescence"]["laser count"]
            lsc2 = 0
            for ii in range(1, 4):
                kl = "laser {} lambda".format(ii)
                kp = "laser {} power".format(ii)
                if (kl in ds.config["fluorescence"] and
                        kp in ds.config["fluorescence"] and
                        ds.config["fluorescence"][kp] != 0):
                    lsc2 += 1
            if lsc1 != lsc2:
                msg = "Metadata: fluorescence laser count inconsistent"
                viol.append(msg)
        # check for samples per event
        if "samples per event" in ds.config["fluorescence"]:
            spe = ds.config["fluorescence"]["samples per event"]
            if "trace" in ds:
                for key in ds["trace"].keys():
                    spek = ds["trace"][key][0].size
                    if spek != spe:
                        msg = "Metadata: wrong number of samples per event: " \
                              + "{} (expected {}, got {})".format(key, spe,
                                                                  spek)
                        viol.append(msg)
        # check for channel names
        for ii in range(1, 4):
            chn = "channel {} name".format(ii)
            fli = "fl{}_max".format(ii)
            if fli in ds and chn not in ds.config["fluorescence"]:
                # Channel names must be defined when there is
                # a corresponding fluorescence signal.
                aler.append("Metadata: Missing key [{}] '{}'".format(
                    "fluorescence", chn))
            elif fli not in ds and chn in ds.config["fluorescence"]:
                # Channel names must not be defined when there is
                # no corresponding fluorescence signal.
                aler.append("Metadata: Unused key defined [{}] '{}'".format(
                    "fluorescence", chn))
    else:
        info.append("Fluorescence: False")
    # search for missing keys (hard)
    for sec in tocheck:
        if sec not in ds.config:
            viol.append("Metadata: Missing section '{}'".format(sec))
        else:
            for key in dfn.config_keys[sec]:
                if (key in tocheck[sec]
                        and key not in ds.config[sec]):
                    viol.append("Metadata: Missing key [{}] '{}'".format(sec,
                                                                         key))
                elif (sec in tocheck_sec_aler
                        and key not in ds.config[sec]):
                    # Note that fluorescence data is not checked here
                    aler.append("Metadata: Missing key [{}] '{}'".format(
                        sec, key))
    # search again (soft)
    for sec in tocheck_sec_aler:
        if sec in tocheck:
            # already treated above (hard)
            continue
        if sec not in ds.config:
            aler.append("Metadata: Missing section '{}'".format(sec))
        else:
            for key in dfn.config_keys[sec]:
                if key not in ds.config[sec]:
                    aler.append("Metadata: Missing key [{}] '{}'".format(sec,
                                                                         key))
    # check for medium
    if "medium" in ds.config["setup"]:
        med = ds.config["setup"]["medium"]
        if med not in ["CellCarrier", "CellCarrierB", "water", "other"]:
            msg = "Metadata: Invalid value [setup] medium: '{}'".format(med)
            viol.append(msg)
    # check for feature column names
    for feat in ds._events.keys():
        if feat not in dfn.feature_names:
            viol.append("Features: Unknown key '{}'".format(feat))
    info.append("Data file format: {}".format(ds.format))
    # hdf5-based checks
    if ds.format == "hdf5":
        # check meta data of images
        if "image" in ds._events:
            imdat = ds["image"]
            for key, val in [['CLASS', b'IMAGE'],
                             ['IMAGE_VERSION', b'1.2'],
                             ['IMAGE_SUBCLASS', b'IMAGE_GRAYSCALE']]:
                if key not in imdat.attrs:
                    aler.append("HDF5: '/image': missing attribute "
                                + "'{}'".format(key))
                elif not isinstance(imdat.attrs[key], bytes):
                    aler.append("HDF5: '/image': attribute '{}' ".format(key)
                                + "should be fixed-length ASCII string")
                elif imdat.attrs[key] != val:
                    aler.append("HDF5: '/image': attribute '{}' ".format(key)
                                + "should have value '{}'".format(val))
        # check length of logs
        with h5py.File(ds.path, mode="r") as h5:
            logs = h5["logs"]
            for logname in logs.keys():
                # ignore tmds meta data log files
                logends = [logname.endswith(l) for l in IGNORED_LOG_NAMES]
                if sum(logends):
                    continue
                log = logs[logname]
                for ii in range(len(log)):
                    if len(log[ii]) > LOG_MAX_LINE_LENGTH:
                        aler.append("Logs: {} line {} ".format(logname, ii)
                                    + "exceeds maximum line length "
                                    + "{}".format(LOG_MAX_LINE_LENGTH))
    # check feature size
    lends = len(ds)
    for feat in ds.features_innate:
        msg = "Features: wrong event count: '{}' ({} of {})"
        if feat == "trace":
            for tr in list(ds["trace"].keys()):
                if len(ds["trace"][tr]) != lends:
                    viol.append(msg.format("trace/".format(tr),
                                           len(ds["trace"][tr]),
                                           lends))
        else:
            if len(ds[feat]) != lends:
                viol.append(msg.format(feat, len(ds[feat]), lends))
    return sorted(viol), sorted(aler), sorted(info)
