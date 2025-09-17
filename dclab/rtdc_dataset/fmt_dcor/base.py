"""DCOR client interface"""
import logging
import pathlib
import re
import time

from ...util import hashobj

from ..config import Configuration
from ..core import RTDCBase
from ..feat_basin import PerishableRecord

from . import api
from .logs import DCORLogs
from .tables import DCORTables


logger = logging.getLogger(__name__)


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
        self._cache_basin_dict = None
        self.cache_basin_dict_time = 600
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

    def _basin_refresh(self, basin):
        """Refresh the specified basin"""
        # Retrieve the basin dictionary from DCOR
        basin_dicts = self.basins_get_dicts()
        for bn_dict in basin_dicts:
            if bn_dict.get("name") == basin.name:
                break
        else:
            raise ValueError(f"Basin '{basin.name}' not found in {self}")

        tre = bn_dict["time_request"]
        ttl = bn_dict["time_expiration"]
        # remember time relative to time.time, subtract 30s to be on safe side
        tex = bn_dict["time_local_request"] + (ttl - tre) - 30

        if isinstance(basin.perishable, bool):
            logger.debug("Initializing basin perishable %s", basin.name)
            # create a perishable record
            basin.perishable = PerishableRecord(
                basin=basin,
                expiration_func=self._basin_expiration,
                expiration_kwargs={"time_local_expiration": tex},
                refresh_func=self._basin_refresh,
            )
        else:
            logger.debug("Refreshing basin perishable %s", basin.name)
            # only update (this also works with weakref.ProxyType)
            basin.perishable.expiration_kwargs = {"time_local_expiration": tex}

        if len(bn_dict["urls"]) > 1:
            logger.warning(f"Basin {basin.name} has multiple URLs. I am not "
                           f"checking their availability: {bn_dict}")
        basin.location = bn_dict["urls"][0]

    def _basin_expiration(self, basin, time_local_expiration):
        """Check whether the basin has perished"""
        return time_local_expiration < time.time()

    def _basins_get_dicts(self):
        try:
            basin_dicts = self.api.get(query="basins")
            # Fill in missing timing information
            for bn_dict in basin_dicts:
                if (bn_dict.get("format") == "http"
                        and "perishable" not in bn_dict):
                    # We are communicating with an older version of
                    # ckanext-dc_serve. Take a look at the URL and check
                    # whether we have a perishable (~1 hour) URL or whether
                    # this is a public resource.
                    expires_regexp = re.compile(".*expires=([0-9]*)$")
                    for url in bn_dict.get("urls", []):
                        if match := expires_regexp.match(url.lower()):
                            logger.debug("Detected perishable basin: %s",
                                         bn_dict["name"])
                            bn_dict["perishable"] = True
                            bn_dict["time_request"] = time.time()
                            bn_dict["time_expiration"] = int(match.group(1))
                            # add part of the resource ID to the name
                            infourl = url.split(bn_dict["name"], 1)[-1]
                            infourl = infourl.replace("/", "")
                            bn_dict["name"] += f"-{infourl[:5]}"
                            break
                    else:
                        bn_dict["perishable"] = False
                # If we have a perishable basin, add the local request time
                if bn_dict.get("perishable"):
                    bn_dict["time_local_request"] = time.time()
        except api.DCORAccessError:
            # TODO: Do not catch this exception when all DCOR instances
            #       implement the 'basins' query.
            # This means that the server does not implement the 'basins' query.
            basin_dicts = []
        return basin_dicts

    def basins_get_dicts(self):
        """Return list of dicts for all basins defined on DCOR

        The return value of this method is cached for 10 minutes
        (cache time defined in the `cache_basin_dict_time` [s] property).
        """
        if (self._cache_basin_dict is None
            or time.time() > (self._cache_basin_dict[1]
                              + self.cache_basin_dict_time)):
            self._cache_basin_dict = (self._basins_get_dicts(), time.time())
        return self._cache_basin_dict[0]

    def basins_retrieve(self):
        """Same as superclass, but add perishable information"""
        basin_dicts = self.basins_get_dicts()
        basins = super(RTDC_DCOR, self).basins_retrieve()
        for bn in basins:
            for bn_dict in basin_dicts:
                if bn.name == bn_dict.get("name"):
                    # Determine whether we have to set a perishable record.
                    if bn_dict.get("perishable"):
                        # required for `_basin_refresh` to create a record
                        bn.perishable = True
                        # create the actual record
                        self._basin_refresh(bn)
                    break
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
