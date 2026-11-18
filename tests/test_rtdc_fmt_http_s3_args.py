"""Test arguments of RTDC_HTTP and RTDC_S3 (without network access)"""
import io

import pytest

import dclab
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


def test_new_dataset_s3_kwargs_use_s3(monkeypatch):
    """S3-specific arguments must not be passed to `RTDC_HTTP` (#266)"""
    pytest.importorskip("boto3")
    path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    url = "https://objectstore.example.com/bucket/resource/abc"
    # the URL is publicly available via HTTP
    monkeypatch.setattr(fmt_http, "is_url_available",
                        lambda *args, **kwargs: True)
    monkeypatch.setattr(fmt_http, "HTTPFile",
                        lambda url: LocalFile(path, url))
    monkeypatch.setattr(fmt_s3, "S3File",
                        lambda object_path, **kwargs: LocalFile(path, url))
    with dclab.new_dataset(url, access_key_id="abc",
                           secret_access_key="def") as ds:
        assert isinstance(ds, fmt_s3.RTDC_S3)


def test_new_dataset_s3_kwargs_without_boto3(monkeypatch):
    """Missing boto3 must result in a helpful error message (#266)"""
    url = "https://objectstore.example.com/bucket/resource/abc"
    monkeypatch.setattr(fmt_http, "is_url_available",
                        lambda *args, **kwargs: True)
    monkeypatch.setattr(fmt_s3, "BOTO3_AVAILABLE", False)
    with pytest.raises(ModuleNotFoundError, match="boto3"):
        dclab.new_dataset(url, access_key_id="abc")


def test_new_dataset_http_without_requests(monkeypatch):
    """Missing requests must result in a helpful error message (#266)"""
    monkeypatch.setattr(fmt_http, "REQUESTS_AVAILABLE", False)
    monkeypatch.setattr(fmt_http, "is_url_available",
                        lambda *args, **kwargs: False)
    with pytest.raises(ModuleNotFoundError, match="requests"):
        dclab.new_dataset("https://example.com/data.rtdc")
