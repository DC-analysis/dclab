import uuid

import pytest


from dclab.http_utils import (
    is_http_url, is_url_available, HTTPFile)

from helper_methods import DCOR_AVAILABLE

pytest.importorskip("requests")

if not DCOR_AVAILABLE:
    pytest.skip("No connection to DCOR", allow_module_level=True)


def test_http_file_basic():
    """This is a basic seek and read test

    It checks whether seeking to a position, reading things and then
    seeking to the same position yields the same result. Since HTTPFile
    caches things, we create two separate HTTPFiles.
    """
    s3_url = ("https://objectstore.hpccloud.mpcdf.mpg.de/"
              "circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
              "resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0")
    f1 = HTTPFile(s3_url)
    f2 = HTTPFile(s3_url)

    f1.seek(5000000)
    d1 = f1.read(200)

    f2.read(200)
    f2.seek(5000050)
    f2.read(200)
    f2.seek(5000000)
    d2 = f2.read(200)
    assert d1 == d2


def test_http_file_does_not_exist():
    bad_s3_url = ("https://objectstore.hpccloud.mpcdf.mpg.de/"
                  "circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
                  "resource/fb7/19f/ef-ffff-ffff-ffff-f4002af916f0")
    f1 = HTTPFile(bad_s3_url)
    with pytest.raises(ValueError,
                       match="Server replied with status code 403 Forbidden"):
        f1._parse_header()


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


@pytest.mark.parametrize("url", [
    "ftp://example.com/bucket/key",  # wrong scheme
    "example.com/bucket/key",  # missing scheme
    "example.com:80",  # missing key
    ])
def test_regexp_http_url_invalid(url):
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
def test_regexp_http_url_valid(url):
    assert is_http_url(url)
