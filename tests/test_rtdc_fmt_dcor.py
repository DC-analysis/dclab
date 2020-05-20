#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test DCOR format"""
from __future__ import print_function, unicode_literals

import socket
import sys

import dclab
from dclab.rtdc_dataset.fmt_dcor import RTDC_DCOR
import numpy as np
import pytest

from helper_methods import retrieve_data, cleanup


if sys.version_info[0] >= 3:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect(("dcor.mpl.mpg.de", 443))
            DCOR_AVAILABLE = True
        except socket.gaierror:
            DCOR_AVAILABLE = False
else:
    # skip test on python2
    DCOR_AVAILABLE = False


class MockAPIHandler(dclab.rtdc_dataset.fmt_dcor.APIHandler):
    def get(self, query, feat=None, trace=None, event=None):
        """Mocks communication with the DCOR API"""
        h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
        with dclab.new_dataset(h5path) as ds:
            if query == "size":
                return len(ds)
            elif query == "metadata":
                return ds.config
            elif query == "feature_list":
                return ds.features
            elif query == "feature" and dclab.dfn.scalar_feature_exists(feat):
                return ds[feat]
            elif query == "trace_list":
                return sorted(ds["trace"].keys())
            elif query == "trace":
                return ds["trace"][trace][event]
            else:
                return ds[feat][event]


def test_dcor_base(monkeypatch):
    monkeypatch.setattr(dclab.rtdc_dataset.fmt_dcor,
                        "APIHandler",
                        MockAPIHandler)
    with dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip")) as ds:
        dso = dclab.new_dataset("https://example.com/api/3/action/dcserv?id=1")
        assert len(dso) == len(ds)
        assert dso.config["setup"]["channel width"] == \
            ds.config["setup"]["channel width"]
        assert np.all(dso["area_um"] == ds["area_um"])
        assert np.all(dso["area_um"] == ds["area_um"])  # test cache
        assert np.all(dso["image"][4] == ds["image"][4])
        assert len(dso["image"]) == len(ds)
        for key in dso._events:
            assert key in ds
        for m, n in zip(dso["mask"], ds["mask"]):
            assert np.all(m == n)
        # compute an ancillary feature
        assert np.all(dso["volume"] == ds["volume"])
        assert np.all(dso["volume"] == ds["volume"])  # test cache
        # trace
        assert sorted(dso["trace"].keys()) == sorted(ds["trace"].keys())
        assert len(dso["trace"]["fl1_raw"]) == len(ds["trace"]["fl1_raw"])
        assert np.all(dso["trace"]["fl1_raw"][1] == ds["trace"]["fl1_raw"][1])
        for t1, t2 in zip(dso["trace"]["fl1_raw"], ds["trace"]["fl1_raw"]):
            assert np.all(t1 == t2)
    cleanup()


@pytest.mark.skipif(not DCOR_AVAILABLE, reason="DCOR not reachable!")
def test_dcor_cache_scalar():
    # calibration beads
    with dclab.new_dataset("fb719fb2-bd9f-817a-7d70-f4002af916f0") as ds:
        # sanity checks
        assert len(ds) == 5000
        assert "area_um" in ds

        area_um = ds["area_um"]
        assert ds["area_um"] is area_um, "Check proper caching"
        # provoke cache deletion
        ds._events._scalar_cache.pop("area_um")
        assert ds["area_um"] is not area_um, "test removal from cache"


@pytest.mark.skipif(not DCOR_AVAILABLE, reason="DCOR not reachable!")
def test_dcor_cache_trace():
    # calibration beads
    with dclab.new_dataset("fb719fb2-bd9f-817a-7d70-f4002af916f0") as ds:
        # sanity checks
        assert len(ds) == 5000
        assert "trace" in ds

        trace0 = ds["trace"]["fl1_raw"][0]
        assert ds["trace"]["fl1_raw"][0] is trace0, "Check proper caching"
        assert ds["trace"]["fl1_raw"][1] is not trace0, "Check proper caching"


def test_dcor_hierarchy(monkeypatch):
    monkeypatch.setattr(dclab.rtdc_dataset.fmt_dcor,
                        "APIHandler",
                        MockAPIHandler)
    dso = dclab.new_dataset("https://example.com/api/3/action/dcserv?id=1")
    dsh = dclab.new_dataset(dso)
    assert np.all(dso["area_um"] == dsh["area_um"])
    cleanup()


def test_url():
    target = "https://example.com/api/3/action/dcserv?id=123456"
    assert RTDC_DCOR.get_full_url(
        url="123456",
        use_ssl=True,
        host="example.com") == target
    assert RTDC_DCOR.get_full_url(
        url="http://example.com/api/3/action/dcserv?id=123456",
        use_ssl=True,
        host="example.com") == target
    assert RTDC_DCOR.get_full_url(
        url="example.com/api/3/action/dcserv?id=123456",
        use_ssl=True,
        host="example.com") == target
    assert RTDC_DCOR.get_full_url(
        url="https://example.com/api/3/action/dcserv?id=123456",
        use_ssl=None,
        host="example.com") == target
    assert RTDC_DCOR.get_full_url(
        url="123456",
        use_ssl=None,
        host="example.com") == target
    target2 = "http://example.com/api/3/action/dcserv?id=123456"
    assert RTDC_DCOR.get_full_url(
        url="example.com/api/3/action/dcserv?id=123456",
        use_ssl=False,
        host="example.com") == target2
    assert RTDC_DCOR.get_full_url(
        url="https://example.com/api/3/action/dcserv?id=123456",
        use_ssl=False,
        host="example.com") == target2
    assert RTDC_DCOR.get_full_url(
        url="http://example.com/api/3/action/dcserv?id=123456",
        use_ssl=None,
        host="example.com") == target2


if __name__ == "__main__":
    # Run all tests
    from inspect import signature
    loc = locals()
    for key in list(loc.keys()):
        if (key.startswith("test_")
            and hasattr(loc[key], "__call__")
                and "monkeypatch" not in signature(loc[key]).parameters):
            loc[key]()
