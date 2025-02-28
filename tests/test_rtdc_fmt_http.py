"""Test HTTP format"""
import time

import numpy as np
import pytest


from dclab.rtdc_dataset.fmt_http import RTDC_HTTP

from helper_methods import DCOR_AVAILABLE


pytest.importorskip("requests")

if not DCOR_AVAILABLE:
    pytest.skip("No connection to DCOR", allow_module_level=True)

# 250209_Blood_2025-02-09_09.46_M003_Reference_dcn_export_28.rtdc
s3_url = ("https://objectstore.hpccloud.mpcdf.mpg.de/"
          "circle-442e6d53-c48b-46eb-873c-4a0f98f3827d/"
          "resource/57e/cde/5d-f896-4599-ba35-d1be7defc6fe")


def test_cache_features():
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
    with RTDC_HTTP(s3_url) as ds:
        # This is the HTTP ETag (https://en.wikipedia.org/wiki/HTTP_ETag)
        # given to this resource by the object store. If the file is
        # re-uploaded, the ETag may change and this test will fail.
        assert ds.identifier == "6dd392feb1aeda7cfb73b4ec76c1fe7c"


@pytest.mark.parametrize("netloc", [
    "objectstore.hpccloud.mpcdf.mpg.de",
    "objectstore.hpccloud.mpcdf.mpg.de:443"
])
def test_netloc_vs_hostname(netloc):
    with RTDC_HTTP(s3_url) as ds:
        assert len(ds) == 28
        assert np.allclose(ds["deform"][0], 0.05335504858810891,
                           rtol=0, atol=1e-7)


def test_open_public_s3_dataset():
    with RTDC_HTTP(s3_url) as ds:
        assert ds.config["experiment"]["sample"] == "Reference"
        assert len(ds) == 28
        assert np.allclose(ds["deform"][0], 0.05335504858810891,
                           atol=0,
                           rtol=1e-5)
