"""Tests for `HTTPFile` with a local HTTP server (no internet required)"""
import hashlib
import http.server
import os
import re
import threading

import pytest

from dclab.http_utils import HTTPFile

requests = pytest.importorskip("requests")


class RangeRequestHandler(http.server.BaseHTTPRequestHandler):
    """Serves `server.data` and supports single-range requests

    The first `server.fail_count` range requests are answered with
    the status code `server.fail_status`. All range headers are
    logged in `server.ranges`.
    """
    def log_message(self, *args):
        pass

    def do_GET(self):
        server = self.server
        data = server.data
        rng = self.headers.get("Range")
        if rng:
            server.ranges.append(rng)
            if server.fail_count > 0:
                server.fail_count -= 1
                self.send_body(server.fail_status,
                               b"<Error><Code>SomeError</Code></Error>")
                return
            start, stop = map(
                int, re.match(r"bytes=(\d+)-(\d+)", rng).groups())
            if start >= len(data) or stop < start:
                self.send_body(416,
                               b"<Error><Code>InvalidRange</Code></Error>")
                return
            self.send_body(206, data[start:stop + 1])
        else:
            self.send_body(200, data)

    def send_body(self, status, body):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        etag = hashlib.md5(self.server.data).hexdigest()
        self.send_header("ETag", f'"{etag}"')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.fixture
def http_server():
    """Local HTTP server serving random data (unique ETag for every test)"""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                             RangeRequestHandler)
    server.data = os.urandom(4000)
    server.ranges = []
    server.fail_count = 0
    server.fail_status = 503
    thread = threading.Thread(target=server.serve_forever,
                              kwargs={"poll_interval": 0.05},
                              daemon=True)
    thread.start()
    server.url = f"http://127.0.0.1:{server.server_address[1]}/data.bin"
    yield server
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("status", [404, 403, 416])
def test_http_file_error_not_returned_as_data(http_server, status):
    """Error responses must raise instead of being returned as file data"""
    http_server.fail_status = status
    http_server.fail_count = 1
    f1 = HTTPFile(http_server.url, chunk_size=1024)
    with pytest.raises(requests.HTTPError, match=str(status)):
        f1.read_range(0, 16)
    # The error must not have been cached, the server is fine again.
    assert f1.read_range(0, 16) == http_server.data[:16]
    f2 = HTTPFile(http_server.url, chunk_size=1024)
    assert f2.read_range(0, 16) == http_server.data[:16]


@pytest.mark.parametrize("status", [429, 500, 503])
def test_http_file_transient_error_retried(http_server, status):
    """Transient errors (e.g. "503 SlowDown" from S3) must be retried"""
    http_server.fail_status = status
    http_server.fail_count = 2
    f1 = HTTPFile(http_server.url, chunk_size=1024)
    assert f1.read_range(0, 16) == http_server.data[:16]
    assert http_server.ranges == ["bytes=0-1023"] * 3


def test_http_file_read_beyond_end_of_file(http_server):
    f1 = HTTPFile(http_server.url, chunk_size=1024)
    f1.seek(len(http_server.data) - 5)
    assert f1.read(100) == http_server.data[-5:]
    assert f1.read_range(len(http_server.data), len(http_server.data) + 10) \
        == b""
    # The 416 error page of the server must never have been requested
    assert http_server.ranges == ["bytes=3072-3999"]


def test_http_file_read_chunk_boundary(http_server):
    """Reading up to a chunk boundary must not download the next chunk"""
    f1 = HTTPFile(http_server.url, chunk_size=1000)
    assert f1.read_range(0, 1000) == http_server.data[:1000]
    assert http_server.ranges == ["bytes=0-999"]
    # end of file is also a chunk boundary
    assert f1.read_range(3500, 4000) == http_server.data[3500:]
    assert http_server.ranges == ["bytes=0-999", "bytes=3000-3999"]
