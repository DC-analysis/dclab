"""DCOR client interface"""
import pathlib
import re

from ...util import hashobj

from ..config import Configuration
from ..core import RTDCBase

from . import api
from .logs import DCORLogs
from .tables import DCORTables


#: Append directories here where dclab should look for certificate bundles
#: for a specific host. The directory should contain files named after the
#: hostname, e.g. "dcor.mpl.mpg.de.cert".
DCOR_CERTS_SEARCH_PATHS = []

#: Regular expression for matching a DCOR resource URL
REGEXP_DCOR_URL = re.compile(
    r"^(https?:\/\/)?"  # scheme
    r"([a-z0-9-\.]*\/?api\/3\/action\/dcserv\?id=)?"  # host with API
    r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")  # id


class RTDC_DCOR(RTDCBase):
    def __init__(self, url, host="dcor.mpl.mpg.de", api_key="",
                 use_ssl=None, cert_path=None, dcserv_api_version=2,
                 *args, **kwargs):
        """Wrap around the DCOR API

        Parameters
        ----------
        url: str
            Full URL or resource identifier; valid values are

            - `<https://dcor.mpl.mpg.de/api/3/action/dcserv?id=
              b1404eb5-f661-4920-be79-5ff4e85915d5>`_
            - dcor.mpl.mpg.de/api/3/action/dcserv?id=b1404eb5-f
              661-4920-be79-5ff4e85915d5
            - b1404eb5-f661-4920-be79-5ff4e85915d5
        host: str
            The default host machine used if the host is not given in `url`
        api_key: str
            API key to access private resources
        use_ssl: bool
            Set this to False to disable SSL (should only be used for
            testing). Defaults to None (does not force SSL if the URL
            starts with "http://").
        cert_path: pathlib.Path
            The (optional) path to a server CA bundle; this should only
            be necessary for DCOR instances in the intranet with a custom
            CA or for certificate pinning.
        dcserv_api_version: int
            Version of the dcserv API to use. In version 0.13.2 of
            ckanext-dc_serve, version 2 was introduced which entails
            serving an S3-basin-only dataset.
        *args:
            Arguments for `RTDCBase`
        **kwargs:
            Keyword arguments for `RTDCBase`

        Attributes
        ----------
        path: str
            Full URL to the DCOR resource
        """
        if not api.REQUESTS_AVAILABLE:
            raise ModuleNotFoundError(
                "Package `requests` required for DCOR format!")

        super(RTDC_DCOR, self).__init__(*args, **kwargs)

        self._hash = None
        self.path = RTDC_DCOR.get_full_url(url, use_ssl, host)

        if cert_path is None:
            cert_path = get_server_cert_path(get_host_from_url(self.path))

        self.api = api.APIHandler(url=self.path,
                                  api_key=api_key,
                                  cert_path=cert_path,
                                  dcserv_api_version=dcserv_api_version)

        # Parse configuration
        self.config = Configuration(cfg=self.api.get(query="metadata"))

        # Lazy logs
        self.logs = DCORLogs(self.api)

        # Lazy tables
        self.tables = DCORTables(self.api)

        # Get size
        size = self.config["experiment"].get("event count")
        if size is None:
            size = int(self.api.get(query="size"))
        self._size = size

        self.title = f"{self.config['experiment']['sample']} - " \
            + f"M{self.config['experiment']['run index']}"

    def __len__(self):
        return self._size

    @property
    def hash(self):
        """Hash value based on file name and content"""
        if self._hash is None:
            tohash = [self.path]
            self._hash = hashobj(tohash)
        return self._hash

    @staticmethod
    def get_full_url(url, use_ssl, host=None):
        """Return the full URL to a DCOR resource

        Parameters
        ----------
        url: str
            Full URL or resource identifier; valid values are

            - https://dcor.mpl.mpg.de/api/3/action/dcserv?id=caab96f6-
              df12-4299-aa2e-089e390aafd5'
            - dcor.mpl.mpg.de/api/3/action/dcserv?id=caab96f6-df12-
              4299-aa2e-089e390aafd5
            - caab96f6-df12-4299-aa2e-089e390aafd5
        use_ssl: bool or None
            Set this to False to disable SSL (should only be used for
            testing). Defaults to None (does not force SSL if the URL
            starts with "http://").
        host: str
            Use this host if it is not specified in `url`
        """
        if use_ssl is None:
            if url.startswith("http://"):
                # user wanted it that way
                scheme = "http"
            else:
                scheme = "https"
        elif use_ssl:
            scheme = "https"
        else:
            scheme = "http"
        if url.count("://"):
            base = url.split("://", 1)[1]
        else:
            base = url
        # determine the api_path and the netloc
        if base.count("/"):
            netloc, api_path = base.split("/", 1)
        else:
            netloc = None  # default to `host`
            api_path = "api/3/action/dcserv?id=" + base
        # remove https from host string (user convenience)
        if host is not None:
            host = host.split("://")[-1]

        netloc = host if netloc is None else netloc
        new_url = f"{scheme}://{netloc}/{api_path}"
        return new_url

    def basins_get_dicts(self):
        """Return list of dicts for all basins defined in `self.h5file`"""
        try:
            basins = self.api.get(query="basins")
        except api.DCORAccessError:
            # TODO: Do not catch this exception when all DCOR instances
            #       implement the 'basins' query.
            # This means that the server does not implement the 'basins' query.
            basins = []
        return basins


def get_host_from_url(url):
    """Extract the hostname from a URL"""
    return url.split("://")[1].split("/")[0]


def get_server_cert_path(host):
    """Return server certificate bundle for DCOR `host`"""

    for path in DCOR_CERTS_SEARCH_PATHS:
        path = pathlib.Path(path)
        cert_path = path / f"{host}.cert"
        if cert_path.exists():
            break
    else:
        # use default certificate bundle
        cert_path = api.requests.certs.where()

    return cert_path


def is_dcor_url(string):
    if not isinstance(string, str):
        return False
    else:
        return REGEXP_DCOR_URL.match(string.strip())
