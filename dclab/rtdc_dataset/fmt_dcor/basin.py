import re

from ..feat_basin import Basin

from .api import REQUESTS_AVAILABLE, APIHandler, DCORAccessError
from .base import RTDC_DCOR


REGEXP_FULL_DCOR_URL = re.compile(
    r"^https?:\/\/"  # scheme
    r"[a-z0-9-\.]*\.[a-z0-9-\.]*\/?api\/3\/action\/dcserv\?id="  # host and API
    r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")  # id


class DCORBasin(Basin):
    basin_format = "dcor"
    basin_type = "remote"

    def __init__(self, *args, **kwargs):
        """Access to private and public DCOR resources

        Since version 2 of the DCOR data API, all feature data are
        accessed via :class:`.HTTPBasin`s on S3. The DCOR basin is just
        a wrapper around those `HTTPBasin`s.

        For private resources, the DCOR format facilitates authentication
        via access tokens. Behind the scenes, DCOR creates a pre-signed
        URL to access private data on an S3 object storage provider.
        Note that you must let dclab know your DCOR access
        token via :func:`.APIHandler.add_api_key` for this to work.

        The `location` must be a full DCOR URL, including the scheme
        and netloc, e.g:

            https://dcor.mpl.mpg.de/api/3/action/dcserv?
            id=b1404eb5-f661-4920-be79-5ff4e85915d5
        """
        self._available_verified = None
        super(DCORBasin, self).__init__(*args, **kwargs)

    def _load_dataset(self, location, **kwargs):
        return RTDC_DCOR(location, **kwargs)

    def is_available(self):
        """Check whether a DCOR resource is available

        Notes
        -----
        - Make sure that your DCOR access token is stored in
          :class:`.APIHandler`. You can add tokens with
          :func:`.APIHandler.add_api_key`.
        """
        with self._av_check_lock:
            if not REQUESTS_AVAILABLE:
                # don't even bother
                self._available_verified = False
            elif not is_full_dcor_url(self.location):
                # not a full DCOR URL
                self._available_verified = False
            if self._available_verified is None:
                api = APIHandler(self.location)
                try:
                    self._available_verified = api.get("valid")
                except DCORAccessError:
                    self._available_verified = False
        return self._available_verified


def is_full_dcor_url(string):
    if not isinstance(string, str):
        return False
    else:
        return REGEXP_FULL_DCOR_URL.match(string.strip())
