#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test tdms file format"""
from __future__ import print_function

import numpy as np
import pytest

from dclab import cli, new_dataset
from helper_methods import retrieve_data, cleanup


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.event_contour.'
                            + 'NoContourDataWarning')
def test_join_tdms():
    path_in = retrieve_data("rtdc_data_shapein_v2.0.1.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    cli.join(path_out=path_out, paths_in=[path_in, path_in])

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert len(dsj)
        assert len(dsj) == 2*len(ds0)
        assert np.all(dsj["circ"][:100] == ds0["circ"][:100])
        assert np.all(dsj["circ"][len(ds0):len(ds0)+100] == ds0["circ"][:100])
        assert set(dsj.features) == set(ds0.features)
    cleanup()


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.event_contour.'
                            + 'NoContourDataWarning')
def test_join_tdms_logs():
    path_in = retrieve_data("rtdc_data_shapein_v2.0.1.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    cli.join(path_out=path_out, paths_in=[path_in, path_in])

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "dclab-join" in dsj.logs
        assert "cfg-#1" in dsj.logs
        assert "software version = ShapeIn 2.0.1" in dsj.logs["cfg-#1"]
        assert ds0.logs
        for key in ds0.logs:
            jkey = "src-#1_" + key
            assert np.all(np.array(ds0.logs[key]) == np.array(dsj.logs[jkey]))
    cleanup()


def test_join_rtdc():
    path_in = retrieve_data("rtdc_data_hdf5_mask_contour.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    cli.join(path_out=path_out, paths_in=[path_in, path_in])
    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "dclab-join" in dsj.logs
        assert len(dsj)
        assert len(dsj) == 2*len(ds0)
        assert np.all(dsj["circ"][:len(ds0)] == ds0["circ"])
        assert np.all(dsj["circ"][len(ds0):] == ds0["circ"])
        assert set(dsj.features) == set(ds0.features)
        assert 'identifier = ZMDD-AcC-8ecba5-cd57e2' in dsj.logs["cfg-#1"]


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.event_contour.'
                            + 'NoContourDataWarning')
def test_tdms2rtdc():
    path_in = retrieve_data("rtdc_data_shapein_v2.0.1.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    cli.tdms2rtdc(path_tdms=path_in,
                  path_rtdc=path_out,
                  compute_features=False)

    with new_dataset(path_out) as ds2, new_dataset(path_in) as ds1:
        assert len(ds2)
        assert set(ds1.features) == set(ds2.features)
        # not all features are computed
        assert set(ds2._events.keys()) < set(ds1.features)
        for feat in ds1:
            assert np.all(ds1[feat] == ds2[feat])

    cleanup()


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.event_contour.'
                            + 'NoContourDataWarning')
def test_tdms2rtdc_features():
    path_in = retrieve_data("rtdc_data_shapein_v2.0.1.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    cli.tdms2rtdc(path_tdms=path_in,
                  path_rtdc=path_out,
                  compute_features=True)

    with new_dataset(path_out) as ds2, new_dataset(path_in) as ds1:
        assert len(ds2)
        assert set(ds1.features) == set(ds2.features)
        # features were computed
        assert set(ds2._events.keys()) == set(ds1.features)
    cleanup()


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
