import functools
import io
import os
import re
import socket
from unittest import mock
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


#: Regular expression for matching a regular HTTP URL
REGEXP_HTTP_URL = re.compile(
    r"^(https?:\/\/)"  # protocol (http or https or omitted)
    r"([a-z0-9-\.]*)(\:[0-9]*)?\/"  # host:port
    r".+"  # path
)


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
        self.session = session_cache.get_session(url)
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


class ResoluteRequestsSessionCache:
    def __init__(self):
        """A multiprocessing-safe cache for requests session objects

        This class implements empty `__getstate__` and `__setstate__`
        methods, so that when used in a multiprocessing context, sessions
        are never mirrored to the subprocesses. Each subprocess creates
        its own sessions.

        Note that only :class:`ResoluteRequestsSession` objects are used,
        which is ideal for the use-case of unstable internet connections.
        """
        #: This dictionary holds all sessions in use by the current process.
        #: Sessions are stored with the host name / netloc as the key.
        self.sessions = {}

    def __getstate__(self):
        """Returns None, so sessions are not pickled into subrpocesses"""
        pass

    def __setstate__(self, state):
        """Does nothing (see `__getstate__`)"""
        pass

    def get_session(self, url: str):
        """Return a requests session for the specified URL

        For each hostname, a different session is returned,
        but for identical hostnames, cached sessions are used.
        """
        urlp = urlparse(url)
        key = urlp.netloc
        if key not in self.sessions:
            self.sessions[key] = ResoluteRequestsSession()
        return self.sessions[key]


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
                # Use `hostname`, not `netloc`, because `netloc` contains
                # the port number which we do not want here.
                s.connect((urlp.hostname, port))
            except (socket.gaierror, OSError):
                reason = "no connection"
            else:
                # Try to access the url
                try:
                    ses = session_cache.get_session(url)
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


#: cache of requests sessions for current process
session_cache = ResoluteRequestsSessionCache()
