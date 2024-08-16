import hashlib

from ..http_utils import HTTPFile, REQUESTS_AVAILABLE, is_url_available
from ..http_utils import is_http_url  # noqa: F401

from .feat_basin import Basin
from .fmt_hdf5 import RTDC_HDF5


class RTDC_HTTP(RTDC_HDF5):
    def __init__(self,
                 url: str,
                 *args, **kwargs):
        """Access RT-DC measurements via HTTP

        This class allows you to open .rtdc files accessible via an
        HTTP URL, for instance files on an S3 object storage or
        figshare download links.

        This is essentially just a wrapper around :class:`.RTDC_HDF5`
        with :class:`.HTTPFile` passing a file object to h5py.

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

        Notes
        -----
        Since this format still requires random access to the file online,
        i.e. not the entire file is downloaded, only parts of it, the
        web server must support range requests.
        """
        if not REQUESTS_AVAILABLE:
            raise ModuleNotFoundError(
                f"Package `requests` required for loading http data '{url}'!")

        self._fhttp = HTTPFile(url)
        if kwargs.get("identifier") is None:
            if self._fhttp.etag is not None:
                # Set the HTTP ETag as the identifier, it doesn't get
                # more unique than that!
                kwargs["identifier"] = self._fhttp.etag
            else:
                # Compute a hash of the first data chunk
                kwargs["identifier"] = hashlib.md5(
                    self._fhttp.get_cache_chunk(0)).hexdigest()

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

    def _load_dataset(self, location, **kwargs):
        h5file = RTDC_HTTP(location, **kwargs)
        return h5file

    def is_available(self):
        """Check for `requests` and object availability

        Caching policy: Once this method returns True, it will always
        return True.
        """
        if self._available_verified is None:
            with self._av_check_lock:
                if not REQUESTS_AVAILABLE:
                    # don't even bother
                    self._available_verified = False
                else:
                    avail, reason = is_url_available(self.location,
                                                     ret_reason=True)
                    if reason in ["forbidden", "not found"]:
                        # we cannot access the URL in the near future
                        self._available_verified = False
                    elif avail:
                        self._available_verified = True
        return self._available_verified
