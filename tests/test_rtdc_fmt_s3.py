"""Test S3 format"""
import time
import uuid

import numpy as np
import pytest


from dclab.rtdc_dataset.fmt_s3 import (
    is_s3_url, is_s3_object_available, RTDC_S3)

from helper_methods import DCOR_AVAILABLE


pytest.importorskip("boto3")


if not DCOR_AVAILABLE:
    pytest.skip("No connection to DCOR", allow_module_level=True)


def test_cache_features():
    # This is the calibration beads measurement.
    # https://dcor.mpl.mpg.de/dataset/figshare-7771184-v2/
    # resource/fb719fb2-bd9f-817a-7d70-f4002af916f0
    s3_url = ("https://objectstore.hpccloud.mpcdf.mpg.de/"
              "circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
              "resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0")

    with RTDC_S3(s3_url) as ds:
        t0 = time.perf_counter()
        _ = ds["deform"][:]
        t1 = time.perf_counter()
        for ii in range(10):
            _ = ds["deform"][:]
        t2 = time.perf_counter()
        assert t2 - t1 < t1 - t0


@pytest.mark.parametrize("url, avail", [
    ("https://objectstore.hpccloud.mpcdf.mpg.de/"
     "circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
     "resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0", True),
    # "noexisting"
    ("https://objectstore.hpccloud.mpcdf.mpg.de/"
     "noexisting-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
     "resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0", False),
    # invalid URL
    ("https://example.com", False),
    # nonexistent host
    (f"http://{uuid.uuid4()}.com/bucket/resource", False),
    (f"https://{uuid.uuid4()}.com/bucket/resource", False),
])
def test_object_available(url, avail):
    act = is_s3_object_available(url)
    assert act == avail


def test_open_public_s3_dataset():
    # This is the calibration beads measurement.
    # https://dcor.mpl.mpg.de/dataset/figshare-7771184-v2/
    # resource/fb719fb2-bd9f-817a-7d70-f4002af916f0
    s3_url = ("https://objectstore.hpccloud.mpcdf.mpg.de/"
              "circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
              "resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0")

    with RTDC_S3(s3_url) as ds:
        assert ds.config["experiment"]["sample"] == "calibration_beads"
        assert len(ds) == 5000
        assert np.allclose(ds["deform"][100], 0.013640802,
                           atol=0,
                           rtol=1e-5)


@pytest.mark.parametrize("url", [
    "ftp://example.com/bucket/key",  # wrong scheme
    "example.com/bucket/key",  # missing scheme
    "https://example.com/bucket",  # missing key
    "https://example.com/bucket/",  # missing key
    "example.com:80",  # missing key and bucket
    ])
def test_regexp_s3_url_invalid(url):
    assert not is_s3_url(url)


@pytest.mark.parametrize("url", [
    "https://example.com/bucket/key",
    "https://example.com/bucket/key2/key3",
    "https://example.com:80/bucket/key",
    "https://example.com:443/bucket/key",
    "http://example.com:80/bucket/key",
    "http://example.com/bucket/key",
    ])
def test_regexp_s3_url_valid(url):
    assert is_s3_url(url)
