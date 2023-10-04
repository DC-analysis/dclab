"""Test S3 format"""
import socket
import time

import dclab
import numpy as np
import pytest


pytest.importorskip("requests")


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        s.connect(("dcor.mpl.mpg.de", 443))
        DCOR_AVAILABLE = True
    except (socket.gaierror, OSError):
        DCOR_AVAILABLE = False


def test_open_public_s3_dataset():
    # This is the calibration beads measurement.
    # https://dcor.mpl.mpg.de/dataset/figshare-7771184-v2/
    # resource/fb719fb2-bd9f-817a-7d70-f4002af916f0
    s3_url = ("https://objectstore.hpccloud.mpcdf.mpg.de/"
              "circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
              "resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0")

    ds = dclab.new_dataset(s3_url)
    assert ds.config["experiment"]["sample"] == "calibration_beads"
    assert len(ds) == 5000
    assert np.allclose(ds["deform"][100], 0.013640802,
                       atol=0,
                       rtol=1e-5)


def test_cache_features():
    # This is the calibration beads measurement.
    # https://dcor.mpl.mpg.de/dataset/figshare-7771184-v2/
    # resource/fb719fb2-bd9f-817a-7d70-f4002af916f0
    s3_url = ("https://objectstore.hpccloud.mpcdf.mpg.de/"
              "circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
              "resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0")

    ds = dclab.new_dataset(s3_url)
    t0 = time.perf_counter()
    _ = ds["deform"]
    t1 = time.perf_counter()
    for ii in range(50):
        _ = ds["deform"]
    t2 = time.perf_counter()
    assert t2-t1 < t1 - t0
