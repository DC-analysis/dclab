"""Test HTTP format"""
import time

import numpy as np
import pytest


from dclab.rtdc_dataset.fmt_http import RTDC_HTTP

from helper_methods import DCOR_AVAILABLE


pytest.importorskip("requests")

if not DCOR_AVAILABLE:
    pytest.skip("No connection to DCOR", allow_module_level=True)


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


@pytest.mark.parametrize("netloc", [
    "objectstore.hpccloud.mpcdf.mpg.de",
    "objectstore.hpccloud.mpcdf.mpg.de:443"
])
def test_netloc_vs_hostname(netloc):
    s3_url = (f"https://{netloc}/"
              f"circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
              f"resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0")
    with RTDC_HTTP(s3_url) as ds:
        assert len(ds) == 5000
        assert np.allclose(ds["deform"][0], 0.009741939,
                           rtol=0, atol=1e-7)


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
