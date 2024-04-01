import sys
import tempfile
import pathlib
import shutil

from dclab import cli, new_dataset, rtdc_dataset

import h5py
import numpy as np
import pytest

from helper_methods import retrieve_data



@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_compressed_split():
    """Make sure the split output data are compressed"""
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.parent

    paths = cli.split(path_in=path_in,
                      path_out=path_out,
                      split_events=3,
                      ret_out_paths=True)

    for pp in paths:
        ic = rtdc_dataset.check.IntegrityChecker(pp)
        ccue = ic.check_compression()[0]
        assert ccue.data["uncompressed"] == 0





@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_split():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    paths = cli.split(path_in=path_in, split_events=3, ret_out_paths=True)
    with new_dataset(path_in) as ds:
        ecount = 0
        for pp in paths:
            with new_dataset(pp) as di:
                for feat in ds.features_scalar:
                    if feat in ["index",
                                "time",  # issue 204
                                ]:
                        continue
                    assert np.all(
                        ds[feat][ecount:ecount + len(di)] == di[feat]), feat
                ecount += len(di)




@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_split_traces():
    path_in = retrieve_data("fmt-hdf5_fl_2018.zip")
    paths = cli.split(path_in=path_in, split_events=3, ret_out_paths=True)
    with new_dataset(path_in) as ds:
        ecount = 0
        for pp in paths:
            with new_dataset(pp) as di:
                for flkey in ds["trace"].keys():
                    trace1 = ds["trace"][flkey][ecount:ecount + len(di)]
                    trace2 = di["trace"][flkey][:]
                    assert len(trace1) == len(trace2)
                    assert np.all(trace1 == trace2), flkey
                ecount += len(di)
