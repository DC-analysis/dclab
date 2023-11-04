"""Test HTTP format"""
import socket
import time
import uuid

import numpy as np
import pytest


from dclab.rtdc_dataset.fmt_http import (
    is_http_url, is_url_available, RTDC_HTTP)


pytest.importorskip("requests")


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        s.connect(("dcor.mpl.mpg.de", 443))
    except (socket.gaierror, OSError):
        pytest.skip("No connection to DCOR",
                    allow_module_level=True)


def test_cache_features():
    # This is the calibration beads measurement.
    # https://dcor.mpl.mpg.de/dataset/figshare-7771184-v2/
    # resource/fb719fb2-bd9f-817a-7d70-f4002af916f0
    s3_url = ("https://objectstore.hpccloud.mpcdf.mpg.de/"
              "circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
              "resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0")

    with RTDC_HTTP(s3_url) as ds:
        t0 = time.perf_counter()
        _ = ds["deform"][:]
        _ = ds["image"][10]
        t1 = time.perf_counter()
        for ii in range(50):
            _ = ds["deform"][:]
            _ = ds["image"][10]
        t2 = time.perf_counter()
        assert t2 - t1 < t1 - t0


def test_identifier():
    s3_url = ("https://objectstore.hpccloud.mpcdf.mpg.de/"
              "circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
              "resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0")

    with RTDC_HTTP(s3_url) as ds:
        # This is the HTTP ETag (https://en.wikipedia.org/wiki/HTTP_ETag)
        # given to this resource by the object store. If the file is
        # re-uploaded, the ETag may change and this test will fail.
        assert ds.identifier == "f0104b0ca2e7d6960189c60fc8b4b986-14"


@pytest.mark.parametrize("url, avail", [
    ("https://objectstore.hpccloud.mpcdf.mpg.de/"
     "circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
     "resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0", (True, "none")),
    # "noexisting"
    ("https://objectstore.hpccloud.mpcdf.mpg.de/"
     "noexisting-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
     "resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0", (False, "not found")),
    # invalid URL
    ("https://example.com", (False, "invalid")),
    # nonexistent host
    (f"http://{uuid.uuid4()}.com/bucket/resource", (False, "no connection")),
    (f"https://{uuid.uuid4()}.com/bucket/resource", (False, "no connection")),
])
def test_object_available(url, avail):
    act = is_url_available(url, ret_reason=True)
    assert act == avail


def test_open_public_s3_dataset():
    # This is the calibration beads measurement.
    # https://dcor.mpl.mpg.de/dataset/figshare-7771184-v2/
    # resource/fb719fb2-bd9f-817a-7d70-f4002af916f0
    s3_url = ("https://objectstore.hpccloud.mpcdf.mpg.de/"
              "circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
              "resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0")

    with RTDC_HTTP(s3_url) as ds:
        assert ds.config["experiment"]["sample"] == "calibration_beads"
        assert len(ds) == 5000
        assert np.allclose(ds["deform"][100], 0.013640802,
                           atol=0,
                           rtol=1e-5)


@pytest.mark.parametrize("url", [
    "ftp://example.com/bucket/key",  # wrong scheme
    "example.com/bucket/key",  # missing scheme
    "example.com:80",  # missing key
    ])
def test_regexp_s3_url_invalid(url):
    assert not is_http_url(url)


@pytest.mark.parametrize("url", [
    "https://example.com/bucket/key",
    "https://example.com/bucket/key2/key3",
    "https://example.com:80/bucket/key",
    "https://example.com:443/bucket/key",
    "http://example.com:80/bucket/key",
    "http://example.com/bucket/key",
    "https://example.com/bucket",
    "https://example.com/bucket/",
])
def test_regexp_s3_url_valid(url):
    assert is_http_url(url)
