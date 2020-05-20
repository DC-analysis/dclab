#!/usr/bin/python
# -*- coding: utf-8 -*-
"""DCOR client interface"""
from __future__ import division, print_function, unicode_literals

import numpy as np

from ..compat import PyImportError, lru_cache
from .. import definitions as dfn
from ..util import hashobj

from .config import Configuration
from .core import RTDCBase

try:
    import requests
except PyImportError:
    REQUESTS_AVAILABLE = False
else:
    REQUESTS_AVAILABLE = True


class DCORAccessError(BaseException):
    pass


class APIHandler(object):
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

    def _get(self, query, feat=None, trace=None, event=None, api_key=""):
        qstr = "&query={}".format(query)
        if feat is not None:
            qstr += "&feature={}".format(feat)
        if trace is not None:
            qstr += "&trace={}".format(trace)
        if event is not None:
            qstr += "&event={}".format(event)
        apicall = self.url + qstr
        req = requests.get(apicall, headers={"Authorization": api_key})
        return req.json()

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


class EventFeature(object):
    """Helper class for accessing non-scalar features event-wise"""

    def __init__(self, feature, api):
        self.identifier = api.url + ":" + feature  # for caching ancillaries
        self.feature = feature
        self.api = api

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    @lru_cache(maxsize=100)
    def __getitem__(self, event):
        data = self.api.get(query="feature", feat=self.feature, event=event)
        return np.asarray(data)

    def __len__(self):
        return int(self.api.get(query="size"))


class EventTrace(object):
    """Helper class for accessing traces event-wise"""

    def __init__(self, trace, api):
        self.identifier = api.url + ":" + trace  # for caching ancillaries
        self.trace = trace
        self.api = api

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    @lru_cache(maxsize=100)
    def __getitem__(self, event):
        data = self.api.get(query="trace", trace=self.trace, event=event)
        return np.asarray(data)

    def __len__(self):
        return int(self.api.get(query="size"))


class EventTraceFeature(object):
    """Helper class for accessing traces"""

    def __init__(self, api):
        self.identifier = api.url + ":traces"
        self.api = api
        self.traces = api.get(query="trace_list")
        self._trace_objects = {}

    def __contains__(self, key):
        return key in self.traces

    def __getitem__(self, trace):
        if trace in self.traces:
            if trace not in self._trace_objects:
                self._trace_objects[trace] = EventTrace(trace, self.api)
            return self._trace_objects[trace]
        else:
            raise KeyError("trace '{}' not found!".format(trace))

    def keys(self):
        return self.traces


class FeatureCache(object):
    """Download and cache (scalar only) features from DCOR"""

    def __init__(self, api):
        self.api = api
        self._features = self.api.get(query="feature_list")
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
        elif key == "trace":
            if "trace" not in self._nonsc_features:
                self._nonsc_features["trace"] = EventTraceFeature(self.api)
            return self._nonsc_features["trace"]
        else:
            if key not in self._nonsc_features:
                self._nonsc_features[key] = EventFeature(key, self.api)
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
            raise PyImportError("Package `requests` required for DCOR format!")

        super(RTDC_DCOR, self).__init__(*args, **kwargs)

        self._hash = None
        self.path = RTDC_DCOR.get_full_url(url, use_ssl, host)
        self.api = APIHandler(url=self.path, api_key=api_key)

        # Parse configuration
        self.config = Configuration(cfg=self.api.get(query="metadata"))

        # Get size
        self._size = int(self.api.get(query="size"))

        # Setup events
        self._events = FeatureCache(self.api)

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
