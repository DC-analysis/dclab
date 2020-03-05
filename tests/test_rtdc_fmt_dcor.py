#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test DCOR format"""
from __future__ import print_function, unicode_literals

import dclab
from dclab.rtdc_dataset.fmt_dcor import RTDC_DCOR
import numpy as np

from helper_methods import retrieve_data, cleanup


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
            elif query == "feature" and feat in dclab.dfn.scalar_feature_names:
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
