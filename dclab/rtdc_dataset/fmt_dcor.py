"""DCOR client interface"""
from functools import lru_cache
import json
import numbers
import time
import uuid

import numpy as np

from .. import definitions as dfn
from ..util import hashobj

from .config import Configuration
from .core import RTDCBase

try:
    import requests
except ModuleNotFoundError:
    REQUESTS_AVAILABLE = False
else:
    REQUESTS_AVAILABLE = True


class DCORAccessError(BaseException):
    pass


class APIHandler:
    """Handles the DCOR api with caching for simple queries"""
    #: these are cached to minimize network usage
    cache_queries = ["metadata", "size", "feature_list", "valid"]
    #: DCOR API Keys in the current session
    api_keys = []

    def __init__(self, url, api_key=""):
        self.url = url
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
        qstr = "&query={}".format(query)
        if feat is not None:
            qstr += "&feature={}".format(feat)
        if trace is not None:
            qstr += "&trace={}".format(trace)
        if event is not None:
            qstr += "&event={}".format(event)
        apicall = self.url + qstr
        for ii in range(retries):
            req = requests.get(apicall, headers={"Authorization": api_key})
            try:
                jreq = req.json()
            except json.decoder.JSONDecodeError:
                time.sleep(0.1)  # wait a bit, maybe the server is overloaded
                continue
            else:
                break
        else:
            raise DCORAccessError("Could not complete query '{}', because "
                                  "the response did not contain any JSON-"
                                  "parseable data. Retried {} times.".format(
                                    apicall, retries))
        return jreq

    def get(self, query, feat=None, trace=None, event=None):
        if query in APIHandler.cache_queries and query in self._cache:
            result = self._cache[query]
        else:
            for api_key in [self.api_key] + APIHandler.api_keys:
                req = self._get(query, feat, trace, event, api_key)
                if req["success"]:
                    self.api_key = api_key  # remember working key
                    break
            else:
                raise DCORAccessError("Cannot access {}: {}".format(
                    query, req["error"]["message"]))
            result = req["result"]
            if query in APIHandler.cache_queries:
                self._cache[query] = result
        return result


class DCORNonScalarFeature:
    """Helper class for accessing non-scalar features"""
    def __init__(self, feat, api, size):
        self.identifier = api.url + ":" + feat  # for caching ancillaries
        self.feat = feat
        self.api = api
        self._size = size

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, event):
        if not isinstance(event, numbers.Integral):
            # slicing!
            indices = np.arange(len(self))[event]
            trace0 = self._get_item(indices[0])
            # determine the correct shape from the first feature
            oshape = [len(indices)] + list(trace0.shape)
            output = np.zeros(oshape, dtype=trace0.dtype)
            # populate the output array
            for ii, evid in enumerate(indices):
                output[ii] = self._get_item(evid)
            return output
        else:
            return self._get_item(event)

    def __len__(self):
        return self._size

    @lru_cache(maxsize=100)
    def _get_item(self, event):
        data = self.api.get(query="feature", feat=self.feat, event=event)
        return np.asarray(data)


class DCORContourFeature(DCORNonScalarFeature):
    """Helper class for accessing contour data"""
    def __getitem__(self, event):
        if not isinstance(event, numbers.Integral):
            # We cannot use the original method, because contours
            # may have different sizes! So we return a list.
            indices = np.arange(len(self))[event]
            output = []
            # populate the output list
            for evid in indices:
                output.append(self._get_item(evid))
            return output
        else:
            return self._get_item(event)


class DCORTraceItem(DCORNonScalarFeature):
    """Helper class for accessing traces"""
    @lru_cache(maxsize=100)
    def _get_item(self, event):
        data = self.api.get(query="trace", trace=self.feat, event=event)
        return np.asarray(data)


class DCORTraceFeature:
    """Helper class for accessing traces"""

    def __init__(self, api, size):
        self.identifier = api.url + ":traces"
        self.api = api
        self._size = size
        self.traces = api.get(query="trace_list")
        self._trace_objects = {}

    def __contains__(self, key):
        return key in self.traces

    def __getitem__(self, trace):
        if trace in self.traces:
            if trace not in self._trace_objects:
                self._trace_objects[trace] = DCORTraceItem(trace, self.api,
                                                           self._size)
            return self._trace_objects[trace]
        else:
            raise KeyError("trace '{}' not found!".format(trace))

    def keys(self):
        return self.traces


class FeatureCache:
    """Download and cache (scalar only) features from DCOR"""

    def __init__(self, api, size):
        self.api = api
        self._features = self.api.get(query="feature_list")
        self._size = size
        self._scalar_cache = {}
        self._nonsc_features = {}

    def __contains__(self, key):
        return key in self._features

    def __getitem__(self, key):
        # user-level checking is done in core.py
        assert dfn.feature_exists(key)
        if key not in self._features:
            raise KeyError("Feature '{}' not found!".format(key))

        if key in self._scalar_cache:
            return self._scalar_cache[key]
        elif dfn.scalar_feature_exists(key):
            # download the feature and cache it
            feat = np.asarray(self.api.get(query="feature", feat=key))
            self._scalar_cache[key] = feat
            return feat
        elif key == "contour":
            if key not in self._nonsc_features:
                self._nonsc_features[key] = DCORContourFeature(key, self.api,
                                                               self._size)
            return self._nonsc_features[key]
        elif key == "trace":
            if "trace" not in self._nonsc_features:
                self._nonsc_features["trace"] = DCORTraceFeature(self.api,
                                                                 self._size)
            return self._nonsc_features["trace"]
        else:
            if key not in self._nonsc_features:
                self._nonsc_features[key] = DCORNonScalarFeature(key, self.api,
                                                                 self._size)
            return self._nonsc_features[key]

    def __iter__(self):
        # dict-like behavior
        for key in self.keys():
            yield key

    def keys(self):
        return self._features


class RTDC_DCOR(RTDCBase):
    def __init__(self, url, use_ssl=None, host="dcor.mpl.mpg.de",
                 api_key="", *args, **kwargs):
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
        use_ssl: bool
            Set this to False to disable SSL (should only be used for
            testing). Defaults to None (does not force SSL if the URL
            starts with "http://").
        host: str
            The host machine (used if the host is not given in `url`)
        api_key: str
            API key to access private resources
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
        self.api = APIHandler(url=self.path, api_key=api_key)

        # Parse configuration
        self.config = Configuration(cfg=self.api.get(query="metadata"))

        # Get size
        self._size = int(self.api.get(query="size"))

        # Setup events
        self._events = FeatureCache(self.api, size=self._size)

        # Override logs property with HDF5 data
        self.logs = {}

        self.title = "{} - M{}".format(self.config["experiment"]["sample"],
                                       self.config["experiment"]["run index"])

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
        new_url = "{}://{}/{}".format(web, host, api)
        return new_url

    @property
    def hash(self):
        """Hash value based on file name and content"""
        if self._hash is None:
            tohash = [self.path]
            self._hash = hashobj(tohash)
        return self._hash


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
