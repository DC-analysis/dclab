import pathlib
import socket
import tempfile
import zipfile

import numpy as np

from dclab.rtdc_dataset import fmt_tdms
from dclab import definitions as dfn
from dclab.rtdc_dataset.fmt_dcor import REQUESTS_AVAILABLE  # noqa: F401
from dclab.rtdc_dataset.fmt_s3 import BOTO3_AVAILABLE  # noqa: F401


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        # dcor.mpl.mpg.de
        s.connect(("130.183.217.150", 443))
    except (socket.gaierror, OSError):
        DCOR_AVAILABLE = False
    else:
        DCOR_AVAILABLE = True


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        # objectstore.hpccloud.mpcdf.mpg.de
        s.connect(("130.183.12.74", 443))
    except (socket.gaierror, OSError):
        DCOR_AVAILABLE = False


def calltracker(func):
    """Decorator to track how many times a function is called"""
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return func(*args, **kwargs)
    wrapped.calls = 0
    return wrapped


def example_data_dict(size=100, keys=["area_um", "deform"]):
    """Example dict with which an RTDCBase can be instantiated.
    """
    ddict = {}
    for ii, key in enumerate(keys):
        if key in ["time", "frame"]:
            val = np.arange(size)
        elif key == "contour":
            cdata = []
            for ss in range(size):
                cont = np.array([5, 5,
                                 5, 6,
                                 5, 7,
                                 6, 7,
                                 7, 7,
                                 7, 6,
                                 7, 5,
                                 6, 5,
                                 5, 5]).reshape(-1, 2)
                cdata.append(cont + ss)
            val = cdata
        elif key in ["image", "image_bg"]:
            imdat = []
            for ss in range(size):
                data = np.arange(10 * 20).reshape(10, 20) + ss
                imdat.append(np.array(data, dtype=np.uint8))
            val = imdat
        elif key == "index":
            # index starts at 1
            val = np.arange(1, size+1)
        elif key == "trace":
            trdata = {}
            kk = 1
            for tr in dfn.FLUOR_TRACES:
                trac = np.arange(100 * size, dtype=np.int16).reshape(size, -1)
                trdata[tr] = trac - kk
                kk += 1
            val = trdata
        else:
            state = np.random.RandomState(size + ii)
            val = state.random_sample(size)

        norm = False
        if key == "area_um":
            norm = 400
        elif key in ["deform", "circ"]:
            norm = .02
        if norm:
            vmin, vmax = val.min(), val.max()
            val = (val - vmin) / (vmax - vmin) * norm

        ddict[key] = val

    return ddict


def find_data(path):
    """Find tdms and rtdc data files in a directory"""
    path = pathlib.Path(path)
    tdmsfiles = fmt_tdms.get_tdms_files(path)
    rtdcfiles = [r for r in path.rglob("*.rtdc") if r.is_file()]
    files = [pathlib.Path(ff) for ff in rtdcfiles + tdmsfiles]
    return files


def retrieve_data(zip_file):
    """Extract contents of data zip file and return data files
    """
    zpath = pathlib.Path(__file__).resolve().parent / "data" / zip_file
    # unpack
    arc = zipfile.ZipFile(str(zpath))

    # extract all files to a temporary directory
    edest = tempfile.mkdtemp(prefix=zpath.name)
    arc.extractall(edest)

    # Load RT-DC dataset
    # find tdms files
    datafiles = find_data(edest)

    if len(datafiles) == 1:
        datafiles = datafiles[0]

    return datafiles
