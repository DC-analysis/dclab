webloc = "https://github.com/ZELLMECHANIK-DRESDEN/RTDCdata/raw/master/"

from os.path import join, exists, basename, dirname, abspath
import numpy as np
import tempfile
import sys
import zipfile

sys.path.insert(0, dirname(dirname(abspath(__file__))))

import dclab


def dl_file(url, dest, chunk_size=6553):
    """
    Download `url` to `dest`.
    """
    import urllib3
    http = urllib3.PoolManager()
    r = http.request('GET', url, preload_content=False)
    with open(dest, 'wb') as out:
        while True:
            data = r.read(chunk_size)
            if data is None or len(data)==0:
                break
            out.write(data)
    r.release_conn()
    

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
    url = join(webloc, zip_file)
    dest = join(tempfile.gettempdir(), zip_file)
    # download
    if not exists(dest):
        dl_file(url, dest)
    # unpack
    arc = zipfile.ZipFile(dest)
    
    # extract all files to a temporary directory
    edest = tempfile.mkdtemp(prefix=basename(zip_file))
    arc.extractall(edest)
    
    ## Load RTDC Data set
    # find tdms files
    tdmsfiles = dclab.GetTDMSFiles(edest)
    
    if len(tdmsfiles) == 1:
        tdmsfiles = tdmsfiles[0]

    return tdmsfiles
    
    
example_data_sets = ["SimpleMeasurement.zip"]