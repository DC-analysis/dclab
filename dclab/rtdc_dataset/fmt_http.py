import functools
import re
import socket
from urllib.parse import urlparse


try:
    from fsspec.implementations.http import HTTPFileSystem
    import requests
except ModuleNotFoundError:
    FSSPEC_AVAILABLE = False
else:
    FSSPEC_AVAILABLE = True


from .feat_basin import Basin
from .fmt_hdf5 import RTDC_HDF5


#: Regular expression for matching a DCOR resource URL
REGEXP_HTTP_URL = re.compile(
    r"^(https?:\/\/)"  # protocol (http or https or omitted)
    r"([a-z0-9-\.]*)(\:[0-9]*)?\/"  # host:port
    r".+"  # path
)


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
        if not FSSPEC_AVAILABLE:
            raise ModuleNotFoundError(
                "Package `fsspec[http]` required for http format!")

        HTTPFileSystem.cachable = False
        self._fs = HTTPFileSystem()
        self._fhttp = self._fs.open(url,
                                    block_size=2**18,
                                    cache_type="readahead")
        # Initialize the HDF5 dataset
        super(RTDC_HTTP, self).__init__(
            h5path=self._fhttp,
            *args,
            **kwargs)
        # Override self.path with the actual HTTP URL
        self.path = url


class HTTPBasin(Basin):
    basin_format = "http"
    basin_type = "remote"

    def __init__(self, *args, **kwargs):
        self._available_verified = False
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
        if not self._available_verified:
            self._available_verified = (
                    FSSPEC_AVAILABLE and is_url_available(self.location))
        return self._available_verified


def is_url_available(url: str):
    """Check whether a URL is available

    Parameters
    ----------
    url: str
        full URL to the object
    """
    avail = False
    if is_http_url(url):
        urlp = urlparse(url)
        # default to https if no scheme or port is specified
        port = urlp.port or (80 if urlp.scheme == "http" else 443)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Try to connect to the host
            try:
                s.connect((urlp.netloc, port))
            except (socket.gaierror, OSError):
                pass
            else:
                # Try to access the url
                try:
                    req = requests.get(url, stream=True)
                    avail = req.ok
                except OSError:
                    pass
    return avail


@functools.lru_cache()
def is_http_url(string):
    """Check whether `string` is a valid URL using regexp"""
    if not isinstance(string, str):
        return False
    else:
        return REGEXP_HTTP_URL.match(string.strip())
