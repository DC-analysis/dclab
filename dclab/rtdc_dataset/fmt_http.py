import functools
import io
from unittest import mock
import os
import re
import socket
from urllib.parse import urlparse
import warnings

import numpy as np

try:
    import requests
except ModuleNotFoundError:
    requests = mock.Mock()
    REQUESTS_AVAILABLE = False
else:
    REQUESTS_AVAILABLE = True


from .feat_basin import Basin
from .fmt_hdf5 import RTDC_HDF5


#: Regular expression for matching a DCOR resource URL
REGEXP_HTTP_URL = re.compile(
    r"^(https?:\/\/)"  # protocol (http or https or omitted)
    r"([a-z0-9-\.]*)(\:[0-9]*)?\/"  # host:port
    r".+"  # path
)


class ResoluteRequestsSession(requests.Session):
    """A session with built-in retry for `get`"""
    def get(self, *args, **kwargs):
        kwargs.setdefault("timeout", 0.5)
        for ii in range(100):
            try:
                resp = super(ResoluteRequestsSession,
                             self).get(*args, **kwargs)
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectTimeout,
                    requests.urllib3.exceptions.ConnectionError,
                    requests.urllib3.exceptions.ReadTimeoutError) as e:
                warnings.warn(f"Encountered {e} for {args} {kwargs}")
                continue
            else:
                break
        else:
            raise requests.exceptions.ReadTimeout(
                f"Resolut sesion failed for {args} and {kwargs}!")
        return resp


class HTTPFile(io.IOBase):
    def __init__(self, url, chunk_size=2**18, keep_chunks=200):
        """Chunk-cached access to a URL supporting range requests

        Range requests (https://en.wikipedia.org/wiki/Byte_serving)
        allow clients to access specific parts of a file via HTTP
        without downloading the entire file.

        This class creates a file-like object from a URL that can
        then be passed on to e.g. h5py for reading. In addition, this
        class keeps a chunk cache of the URL, making it (A) fast to
        access frequently used parts of the file and (B) fast to slice
        through large files since the ratio of data downloaded versus
        (time-consuming) HTTP requests is very large.

        Parameters
        ----------
        url: str
            Path to the URL
        chunk_size: int
            Download chunk size. The entire file is split up into
            equally-sized (and thus indexable) chunks.
        keep_chunks: int
            Number of downloaded chunks to keep in memory. For a
            `chunk_size` of 2**18 bytes, a `keep_chunks` of 200
            impliese a chunk cache size of 50 MiB.
        """
        self.url = url
        self._chunk_size = chunk_size
        self._keep_chunks = keep_chunks
        self.session = ResoluteRequestsSession()
        self._len = None
        self._etag = None
        self._pos = 0
        self.cache = {}

    def _parse_header(self):
        """parse the header sent by the server, populates length and etag"""
        if self._len is None:
            # Do not use `self.session.head`, because it might return a
            # wrong content-length for pre-signed S3 URLS.
            resp = self.session.get(self.url, stream=True, timeout=0.5)
            self._len = int(resp.headers["content-length"])
            self._etag = resp.headers.get("etag").strip("'").strip('"')

    @property
    def etag(self):
        """Unique identifier for this resource (version) by the web server"""
        self._parse_header()
        return self._etag

    @property
    def length(self):
        self._parse_header()
        return self._len

    @property
    def max_cache_size(self):
        """The maximum cache size allowed by `chunk_size` and `keep_chunks`"""
        return self._chunk_size * self._keep_chunks

    def close(self):
        """Close the file

        This closes the requests session and then calls `close` on
        the super class.
        """
        self.session.close()
        super(HTTPFile, self).close()

    def download_range(self, start, stop):
        """Download bytes given by the range (`start`, `stop`)

        `stop` is not inclusive (In the HTTP range request it normally is).
        """
        resp = self.session.get(self.url,
                                headers={"Range": f"bytes={start}-{stop-1}"}
                                )
        return resp.content

    def get_cache_chunk(self, index):
        """Return the cache chunk defined by `index`

        If the chunk is not in `self.cache`, it is downloaded.
        """
        if index not in self.cache:
            start = index*self._chunk_size
            stop = min((index+1)*self._chunk_size, self.length)
            self.cache[index] = self.download_range(start, stop)
        if len(self.cache) > self._keep_chunks:
            for kk in self.cache.keys():
                if kk != 0:  # always keep the first chunk
                    self.cache.pop(kk)
                    break
        return self.cache[index]

    def read(self, size=-1, /):
        """Cache-supported read operation (file object)"""
        data = self.read_range_cached(self._pos, self._pos + size)
        if size > 0:
            self._pos += size
        else:
            self._pos = self.length
        return data

    def read_range_cached(self, start, stop):
        """Concatenate the requested bytes from the cached chunks

        This calls `get_cache_chunk` and thus downloads cache
        chunks when necessary.
        """
        toread = stop - start
        # compute the chunk indices between start and stop
        chunk_start = np.int64(start // self._chunk_size)
        chunk_stop = np.int64(stop // self._chunk_size + 1)
        data = b""
        pos = start
        for chunk_index in range(chunk_start, chunk_stop):
            chunk = self.get_cache_chunk(chunk_index)
            chunk_start = pos % self._chunk_size
            if toread == 0:
                break
            elif chunk_start + toread >= self._chunk_size:
                data += chunk[chunk_start:]
                chunks_read = self._chunk_size - chunk_start
            else:
                chunk_end = stop % self._chunk_size
                data += chunk[chunk_start:chunk_end]
                chunks_read = chunk_end - chunk_start
            toread -= chunks_read
            pos += chunks_read

        return data

    def seek(self, offset, whence=os.SEEK_SET):
        """Seek to a position (file object)"""
        if whence == os.SEEK_SET:
            self._pos = offset
        elif whence == os.SEEK_CUR:
            self._pos += offset
        elif whence == os.SEEK_END:
            self._pos = self.length + offset

    def seekable(self):
        """The HTTP file is seekable"""
        return True

    def tell(self):
        """Tell the position (file object)"""
        return self._pos


class RTDC_HTTP(RTDC_HDF5):
    def __init__(self,
                 url: str,
                 *args, **kwargs):
        """Access RT-DC measurements via HTTP

        This is essentially just a wrapper around :class:`.RTDC_HDF5`
        with `fsspec` passing a file object to h5py.

        Parameters
        ----------
        url: str
            Full URL to an HDF5 file
        *args:
            Arguments for `RTDCBase`
        **kwargs:
            Keyword arguments for `RTDCBase`

        Attributes
        ----------
        path: str
            The URL to the object
        """
        if not REQUESTS_AVAILABLE:
            raise ModuleNotFoundError(
                "Package `requests` required for http format!")

        self._fhttp = HTTPFile(url)
        if kwargs.get("identifier") is None:
            # Set the HTTP ETag as the identifier, it doesn't get more unique
            # than that!
            kwargs["identifier"] = self._fhttp.etag
        # Initialize the HDF5 dataset
        super(RTDC_HTTP, self).__init__(
            h5path=self._fhttp,
            *args,
            **kwargs)
        # Override self.path with the actual HTTP URL
        self.path = url

    def close(self):
        super(RTDC_HTTP, self).close()
        self._fhttp.close()


class HTTPBasin(Basin):
    basin_format = "http"
    basin_type = "remote"

    def __init__(self, *args, **kwargs):
        self._available_verified = None
        super(HTTPBasin, self).__init__(*args, **kwargs)

    def load_dataset(self, location, **kwargs):
        h5file = RTDC_HTTP(location, enable_basins=False, **kwargs)
        # If the user specified the events of the basin, then store it
        # directly in the .H5Events class of .RTDC_HDF5. This saves us
        # approximately 2 seconds of listing the members of the "events"
        # group from the URL.
        h5file._events._features_list = self._features
        return h5file

    def is_available(self):
        """Check for fsspec and object availability

        Caching policy: Once this method returns True, it will always
        return True.
        """
        if not REQUESTS_AVAILABLE:
            # don't even bother
            self._available_verified = False
        if self._available_verified is None:
            avail, reason = is_url_available(self.location, ret_reason=True)
            if reason in ["forbidden", "not found"]:
                # we cannot access the URL in the near future
                self._available_verified = False
            elif avail:
                self._available_verified = True
        return self._available_verified


def is_url_available(url: str, ret_reason=False):
    """Check whether a URL is available

    Parameters
    ----------
    url: str
        full URL to the object
    ret_reason: bool
        whether to return reason for unavailability

    Returns
    -------
    available: bool
        whether the URL is available
    reason: str
        reason for the URL not being available is `available` is False
    """
    avail = False
    reason = "none"
    if is_http_url(url):
        urlp = urlparse(url)
        # default to https if no scheme or port is specified
        port = urlp.port or (80 if urlp.scheme == "http" else 443)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            # Try to connect to the host
            try:
                s.connect((urlp.netloc, port))
            except (socket.gaierror, OSError):
                reason = "no connection"
            else:
                # Try to access the url
                try:
                    ses = ResoluteRequestsSession()
                    req = ses.get(url, stream=True, timeout=1)
                    avail = req.ok
                    if not avail:
                        reason = req.reason.lower()
                except OSError:
                    reason = "oserror"
                    pass
    else:
        reason = "invalid"
    if ret_reason:
        return avail, reason
    else:
        return avail


@functools.lru_cache()
def is_http_url(string):
    """Check whether `string` is a valid URL using regexp"""
    if not isinstance(string, str):
        return False
    else:
        return REGEXP_HTTP_URL.match(string.strip())
