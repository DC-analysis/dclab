import socket
import uuid

import pytest


from dclab.http_utils import (
    is_http_url, is_url_available, HTTPFile)


pytest.importorskip("requests")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        s.connect(("dcor.mpl.mpg.de", 443))
    except (socket.gaierror, OSError):
        pytest.skip("No connection to DCOR",
                    allow_module_level=True)


def test_http_file_basic():
    """The is a basic seek and read test

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
