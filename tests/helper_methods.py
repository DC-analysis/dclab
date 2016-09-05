from os.path import join, basename, dirname, abspath
import numpy as np
import tempfile
import sys
import zipfile

sys.path.insert(0, dirname(dirname(abspath(__file__))))

import dclab


def example_data_dict(size=100, keys=["Area", "Defo"]):
    """ Example dict with which an RTDC_DataSet can be instantiated.
    """
    ddict = {}
    for ii, key in enumerate(keys):
        if key in ["Time", "Frame"]:
            val = np.arange(size)
        else:
            state = np.random.RandomState(size+ii)
            val = state.random_sample(size)
        ddict[key] = val
    
    return ddict


def retreive_tdms(zip_file):
    """ Retrieve a zip file that is reachable via the location
    `webloc`, extract it, and return the paths to extracted
    tdms files.
    """
    thisdir = dirname(abspath(__file__))
    ddir = join(thisdir, "data")
    # unpack
    arc = zipfile.ZipFile(join(ddir, zip_file))
    
    # extract all files to a temporary directory
    edest = tempfile.mkdtemp(prefix=basename(zip_file))
    arc.extractall(edest)
    
    ## Load RTDC Data set
    # find tdms files
    tdmsfiles = dclab.GetTDMSFiles(edest)
    
    if len(tdmsfiles) == 1:
        tdmsfiles = tdmsfiles[0]

    return tdmsfiles
    
# Do not change order:    
example_data_sets = ["rtdc_data_minimal.zip",
                     "rtdc_data_traces_video.zip"]
