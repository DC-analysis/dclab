from ..http_utils import HTTPFile, REQUESTS_AVAILABLE, is_url_available
from ..http_utils import is_http_url  # noqa: F401

from .feat_basin import Basin
from .fmt_hdf5 import RTDC_HDF5


class RTDC_HTTP(RTDC_HDF5):
    def __init__(self,
                 url: str,
                 *args, **kwargs):
        """Access RT-DC measurements via HTTP

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
        """Check for `requests` and object availability

        Caching policy: Once this method returns True, it will always
        return True.
        """
        with self._av_check_lock:
            if not REQUESTS_AVAILABLE:
                # don't even bother
                self._available_verified = False
            if self._available_verified is None:
                avail, reason = is_url_available(self.location,
                                                 ret_reason=True)
                if reason in ["forbidden", "not found"]:
                    # we cannot access the URL in the near future
                    self._available_verified = False
                elif avail:
                    self._available_verified = True
        return self._available_verified
