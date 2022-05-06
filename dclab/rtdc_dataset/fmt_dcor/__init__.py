"""DCOR client interface"""
import json
import pathlib
import time
import uuid

from ...util import hashobj

from ..config import Configuration
from ..core import RTDCBase

from .features import FeatureCache


try:
    import requests
except ModuleNotFoundError:
    REQUESTS_AVAILABLE = False
else:
    REQUESTS_AVAILABLE = True

#: Append directories here where dclab should look for certificate bundles
#: for a specific host. The directory should contain files named after the
#: hostname, e.g. "dcor.mpl.mpg.de.cert".
DCOR_CERTS_SEARCH_PATHS = []


class DCORAccessError(BaseException):
    pass


class APIHandler:
    """Handles the DCOR api with caching for simple queries"""
    #: these are cached to minimize network usage
    cache_queries = ["metadata", "size", "feature_list", "valid"]
    #: DCOR API Keys in the current session
    api_keys = []

    def __init__(self, url, api_key="", cert_path=None):
        """

        Parameters
        ----------
        url: str
            URL to DCOR API
        api_key: str
            DCOR API token
        cert_path: pathlib.Path
            the path to the server's CA bundle; by default this
            will use the default certificates (which depends on
            from where you obtained certifi/requests)
        """
        #: DCOR API URL
        self.url = url
        #: keyword argument to :func:`requests.request`
        self.verify = cert_path or True
        #: DCOR API token
        self.api_key = api_key
        self._cache = {}

    @classmethod
    def add_api_key(cls, api_key):
        """Add an API Key to the base class

        When accessing the DCOR API, all available API Keys are
        used to access a resource (trial and error).
        """
        if api_key.strip() and api_key not in APIHandler.api_keys:
            APIHandler.api_keys.append(api_key)

    def _get(self, query, feat=None, trace=None, event=None, api_key="",
             retries=3):
        qstr = f"&query={query}"
        if feat is not None:
            qstr += f"&feature={feat}"
        if trace is not None:
            qstr += f"&trace={trace}"
        if event is not None:
            qstr += f"&event={event}"
        apicall = self.url + qstr
        for _ in range(retries):
            req = requests.get(apicall,
                               headers={"Authorization": api_key},
                               verify=self.verify)
            try:
                jreq = req.json()
            except json.decoder.JSONDecodeError:
                time.sleep(0.1)  # wait a bit, maybe the server is overloaded
                continue
            else:
                break
        else:
            raise DCORAccessError(f"Could not complete query '{apicall}', "
                                  "because the response did not contain any "
                                  f"JSON-parseable data. Retried {retries} "
                                  "times.")
        return jreq

    def get(self, query, feat=None, trace=None, event=None):
        if query in APIHandler.cache_queries and query in self._cache:
            result = self._cache[query]
        else:
            req = {"error": {"message": "No access to API (api key?)"}}
            for api_key in [self.api_key] + APIHandler.api_keys:
                req = self._get(query, feat, trace, event, api_key)
                if req["success"]:
                    self.api_key = api_key  # remember working key
                    break
            else:
                raise DCORAccessError(
                    f"Cannot access {query}: {req['error']['message']}")
            result = req["result"]
            if query in APIHandler.cache_queries:
                self._cache[query] = result
        return result


class RTDC_DCOR(RTDCBase):
    def __init__(self, url, host="dcor.mpl.mpg.de", api_key="",
                 use_ssl=None, cert_path=None, *args, **kwargs):
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
            The host machine (used if the host is not given in `url`)
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
        *args:
            Arguments for `RTDCBase`
        **kwargs:
            Keyword arguments for `RTDCBase`

        Attributes
        ----------
        path: str
            Full URL to the DCOR resource
        """
        if not REQUESTS_AVAILABLE:
            raise ModuleNotFoundError(
                "Package `requests` required for DCOR format!")

        super(RTDC_DCOR, self).__init__(*args, **kwargs)

        self._hash = None
        self.path = RTDC_DCOR.get_full_url(url, use_ssl, host)

        if cert_path is None:
            cert_path = get_server_cert_path(get_host_from_url(self.path))

        self.api = APIHandler(url=self.path, api_key=api_key,
                              cert_path=cert_path)

        # Parse configuration
        self.config = Configuration(cfg=self.api.get(query="metadata"))

        # Get size
        self._size = int(self.api.get(query="size"))

        # Setup events
        self._events = FeatureCache(self.api, size=self._size)

        # Override logs property with HDF5 data
        self.logs = {}

        self.title = f"{self.config['experiment']['sample']} - " \
            + f"M{self.config['experiment']['run index']}"

        # Set up filtering
        self._init_filters()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        pass

    def __len__(self):
        return self._size

    @staticmethod
    def get_full_url(url, use_ssl, host):
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
        use_ssl: bool
            Set this to False to disable SSL (should only be used for
            testing). Defaults to None (does not force SSL if the URL
            starts with "http://").
        host: str
            Use this host if it is not specified in `url`
        """
        if use_ssl is None:
            if url.startswith("http://"):
                # user wanted it that way
                web = "http"
            else:
                web = "https"
        elif use_ssl:
            web = "https"
        else:
            web = "http"
        if url.count("://"):
            base = url.split("://", 1)[1]
        else:
            base = url
        if base.count("/"):
            host, api = base.split("/", 1)
        else:
            api = "api/3/action/dcserv?id=" + base
        new_url = f"{web}://{host}/{api}"
        return new_url

    @property
    def hash(self):
        """Hash value based on file name and content"""
        if self._hash is None:
            tohash = [self.path]
            self._hash = hashobj(tohash)
        return self._hash


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
        cert_path = requests.certs.where()

    return cert_path


def is_dcor_url(string):
    if not isinstance(string, str):
        return False
    elif is_uuid(string):
        return True
    else:  # we have a string
        if string.startswith("http://") or string.startswith("https://"):
            return True  # pretty safe bet
        elif string.count("/api/3/action/dcserv?id="):
            return True  # not so safe, but highly improbable folder structure
        else:
            return False


def is_uuid(string):
    try:
        uuid_obj = uuid.UUID(string)
    except ValueError:
        return False
    return str(uuid_obj) == string
