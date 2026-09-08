from __future__ import annotations

import hashlib
import json
import numbers
import pathlib
import platform
import time
import warnings

import h5py
import numpy as np

try:
    import imageio
except ModuleNotFoundError:
    imageio = None

try:
    import nptdms
except ModuleNotFoundError:
    nptdms = None

from ..rtdc_dataset import fmt_tdms
from .. import util
from .._version import version


class ExtendedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, pathlib.Path):
            return str(o)
        elif isinstance(o, numbers.Integral):
            return int(o)
        elif isinstance(o, np.bool_):
            return bool(o)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)


def assemble_warnings(w):
    """pretty-format all warnings for logs"""
    wlog = []
    for wi in w:
        wlog.append(f"{wi.category.__name__}")
        wlog.append(f" in {wi.category.__module__} line {wi.lineno}:")
        # make sure line is not longer than 100 characters
        words = str(wi.message).split(" ")
        wline = "  "
        for ii in range(len(words)):
            wline += " " + words[ii]
            if ii == len(words) - 1:
                # end
                wlog.append(wline)
            elif len(wline + words[ii+1]) + 1 >= 100:
                # new line
                wlog.append(wline)
                wline = "  "
            # nothing to do here
    return wlog


def get_command_log(paths, custom_dict=None):
    """Return a json dump of system parameters

    Parameters
    ----------
    paths: list of pathlib.Path or str
        paths of related measurement files; up to 5MB of each of
        them is md5-hashed and included in the "files" key
    custom_dict: dict
        additional user-defined entries; must contain simple
        Python objects (json.dumps must still work)
    """
    if custom_dict is None:
        custom_dict = {}
    data = get_job_info()
    data["files"] = []
    for ii, pp in enumerate(paths):
        if pathlib.Path(pp).exists():
            fdict = {"name": pathlib.Path(pp).name,
                     # Hash only 5 MB of the input file
                     "md5-5M": util.hashfile(pp,
                                             count=80,
                                             constructor=hashlib.md5),
                     "index": ii+1
                     }
        else:
            fdict = {"name": f"{pp}",
                     "index": ii + 1
                     }
        data["files"].append(fdict)
    final_data = {}
    final_data.update(custom_dict)
    final_data.update(data)
    dump = json.dumps(final_data,
                      sort_keys=True,
                      indent=2,
                      cls=ExtendedJSONEncoder).split("\n")
    return dump


def get_job_info():
    """Return dictionary with current job information

    Returns
    -------
    info: dict of dicts
        Job information including details about time, system,
        python version, and libraries used.
    """
    data = {
        "utc": {
            "date": time.strftime("%Y-%m-%d", time.gmtime()),
            "time": time.strftime("%H:%M:%S", time.gmtime()),
        },
        "system": {
            "info": platform.platform(),
            "machine": platform.machine(),
            "name": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "python": {
            "build": ", ".join(platform.python_build()),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "libraries": {
            "dclab": version,
            "h5py": h5py.__version__,
            "numpy": np.__version__,
        }
    }
    if imageio is not None:
        data["libraries"]["imageio"] = imageio.__version__
    if nptdms is not None:
        data["libraries"]["nptdms"] = nptdms.__version__
    return data


def monitor(task, bytes_total, bytes_written, stop_event):
    prev_progress = ""

    while not stop_event.is_set():
        if bytes_total.value == 0:
            frac = 0
        else:
            frac = bytes_written.value / bytes_total.value
        progress = f"{task} {frac:.0%}"
        if progress != prev_progress:
            prev_progress = progress
            print(progress, end="\r", flush=True)
        time.sleep(.25)

    if bytes_written.value == bytes_total.value != 0:
        print(f"{task} 100%")
    else:
        print()


def print_info(string):
    print(f"\033[1m{string}\033[0m")


def print_alert(string):
    print_info(f"\033[33m{string}")


def print_violation(string):
    print_info(f"\033[31m{string}")


def skip_empty_image_events(ds, initial=True, final=True):
    """Set a manual filter to skip inital or final empty image events"""
    if initial:
        if (("image" in ds and ds.format == "tdms"
             and ds.config["fmt_tdms"]["video frame offset"])
            or ("contour" in ds and np.all(ds["contour"][0] == 0))
                or ("image" in ds and np.all(ds["image"][0] == 0))):
            ds.filter.manual[0] = False
            ds.apply_filter()
    if final:
        # This is not easy to test, because we need a corrupt
        # frame.
        if "image" in ds:
            idfin = len(ds) - 1
            if ds.format == "tdms":
                with warnings.catch_warnings(record=True) as wfin:
                    warnings.simplefilter(
                        "always",
                        fmt_tdms.event_image.CorruptFrameWarning)
                    _ = ds["image"][idfin]  # provoke a warning
                for ww in wfin:
                    if ww.category == fmt_tdms.event_image.CorruptFrameWarning:
                        ds.filter.manual[idfin] = False
                        ds.apply_filter()
                        break
            elif np.all(ds["image"][idfin] == 0):
                ds.filter.manual[idfin] = False
                ds.apply_filter()
