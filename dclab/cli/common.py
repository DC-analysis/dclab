import hashlib
import json
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
        paths of related measurement files; they are hashed
        and included in the "files" key
    custom_dict: dict
        additional user-defined entries; must contain simple
        Python objects (json.dumps must still work)
    """
    if custom_dict is None:
        custom_dict = {}
    data = get_job_info()
    data["files"] = []
    for ii, pp in enumerate(paths):
        fdict = {"name": pathlib.Path(pp).name,
                 "sha256": util.hashfile(pp, constructor=hashlib.sha256),
                 "index": ii+1
                 }
        data["files"].append(fdict)
    final_data = {}
    final_data.update(custom_dict)
    final_data.update(data)
    dump = json.dumps(final_data, sort_keys=True, indent=2).split("\n")
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


def print_info(string):
    print(f"\033[1m{string}\033[0m")


def print_alert(string):
    print_info(f"\033[33m{string}")


def print_violation(string):
    print_info(f"\033[31m{string}")


def setup_task_paths(paths_in, paths_out, allowed_input_suffixes):
    """Setup directories for a CLI task

    Parameters
    ----------
    paths_in: list of str or lsit of pathlib.Path or str or pathlib.Path
        Input paths
    paths_out: list of str or list of pathlib.Path or str or pathlib.Path
        Output paths
    allowed_input_suffixes: list
        List of allowed input suffixes (e.g. [".rtdc"])

    Returns
    -------
    paths_in: list of pathlib.Path or pathlib.Path
        Input paths
    paths_out: list of pathlib.Path or pathlib.Path
        Output paths
    paths_temp: list of pathlib.Path or pathlib.Path
        Temporary paths (working path)
    """
    if isinstance(paths_in, list):
        list_in = True
    else:
        paths_in = [paths_in]
        list_in = False

    if isinstance(paths_out, list):
        list_out = True
    else:
        paths_out = [paths_out]
        list_out = False

    paths_in = [pathlib.Path(pi) for pi in paths_in]
    for pi in paths_in:
        if pi.suffix not in allowed_input_suffixes:
            raise ValueError(f"Unsupported file type: '{pi.suffix}'")

    paths_out = [pathlib.Path(po) for po in paths_out]
    for ii, po in enumerate(paths_out):
        if po.suffix != ".rtdc":
            paths_out[ii] = po.with_name(po.name + ".rtdc")
    [po.unlink() for po in paths_out if po.exists()]

    paths_temp = [po.with_suffix(".rtdc~") for po in paths_out]
    [pt.unlink() for pt in paths_temp if pt.exists()]

    # convert lists back to paths
    if not list_in:
        paths_in = paths_in[0]

    if not list_out:
        paths_out = paths_out[0]
        paths_temp = paths_temp[0]

    return paths_in, paths_out, paths_temp


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
                    if wfin:
                        ds.filter.manual[idfin] = False
                        ds.apply_filter()
            elif np.all(ds["image"][idfin] == 0):
                ds.filter.manual[idfin] = False
                ds.apply_filter()
