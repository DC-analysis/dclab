"""Test dclab.rtdc_dataset.load"""
import io

import h5py

import dclab
from dclab.rtdc_dataset import fmt_http, rtdc_copy

from helper_methods import retrieve_data


def test_new_dataset_bytesio_identifier():
    """The identifier must be passed on for file objects"""
    path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    fd = io.BytesIO()
    with h5py.File(fd, "w") as hio, h5py.File(path) as h5:
        rtdc_copy(src_h5file=h5, dst_h5file=hio)

    with dclab.new_dataset(fd, identifier="peter") as ds:
        assert ds.identifier == "peter"


def test_new_dataset_http_kwargs(monkeypatch):
    """Keyword arguments must be passed on to RTDC_HTTP"""
    received = {}

    class MockRTDC_HTTP:
        def __init__(self, url, **kwargs):
            received.update(kwargs)

    monkeypatch.setattr(fmt_http, "is_url_available",
                        lambda *args, **kwargs: True)
    monkeypatch.setattr(fmt_http, "RTDC_HTTP", MockRTDC_HTTP)

    dclab.new_dataset("https://example.com/data.rtdc",
                      identifier="peter",
                      enable_basins=False)
    assert received["identifier"] == "peter"
    assert received["enable_basins"] is False
