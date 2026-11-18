"""Test arguments of RTDC_HTTP and RTDC_S3 (without network access)"""
import io

import pytest

from dclab.rtdc_dataset import fmt_http, fmt_s3

from helper_methods import retrieve_data


class LocalFile(io.FileIO):
    """Local file object with the attributes of `HTTPFile`/`S3File`"""
    def __init__(self, path, url):
        super().__init__(path, "r")
        self.url = url
        self.etag = None


def test_http_positional_args(monkeypatch):
    """Positional arguments must be passed on to `RTDC_HDF5`"""
    pytest.importorskip("requests")
    path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    url = "https://example.invalid/data.rtdc"
    monkeypatch.setattr(fmt_http, "HTTPFile",
                        lambda url: LocalFile(path, url))
    # `h5kwargs` as positional argument
    with fmt_http.RTDC_HTTP(url, {"rdcc_nbytes": 1024}) as ds:
        assert len(ds) > 0
        assert ds.path == url


def test_s3_positional_args(monkeypatch):
    """Positional arguments must be passed on to `RTDC_HDF5`"""
    pytest.importorskip("boto3")
    path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    url = "https://example.invalid/bucket/data.rtdc"
    monkeypatch.setattr(fmt_s3, "S3File",
                        lambda object_path, **kwargs: LocalFile(path, url))
    # `h5kwargs` as positional argument (after `use_ssl`)
    with fmt_s3.RTDC_S3(url, None, None, None, True,
                        {"rdcc_nbytes": 1024}) as ds:
        assert len(ds) > 0
        assert ds.path == url
