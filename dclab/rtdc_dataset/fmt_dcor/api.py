import json
import time

from ...http_utils import REQUESTS_AVAILABLE  # noqa: F401
from ...http_utils import requests, session_cache


class DCORAccessError(BaseException):
    pass


class APIHandler:
    """Handles the DCOR api with caching for simple queries"""
    #: these are cached to minimize network usage
    cache_queries = ["metadata", "size", "feature_list", "valid"]
    #: DCOR API Keys/Tokens in the current session
    api_keys = []

    def __init__(self, url, api_key="", cert_path=None, dcserv_api_version=2):
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
        #: DCOR API Token
        self.api_key = api_key
        #: ckanext-dc_serve dcserv API version
        self.dcserv_api_version = dcserv_api_version
        #: create a session
        self.session = session_cache.get_session(url)
        self._cache = {}

    @classmethod
    def add_api_key(cls, api_key):
        """Add an API Key/Token to the base class

        When accessing the DCOR API, all available API Keys/Tokens are
        used to access a resource (trial and error).
        """
        if api_key.strip() and api_key not in APIHandler.api_keys:
            APIHandler.api_keys.append(api_key)

    def _get(self, query, feat=None, trace=None, event=None, api_key="",
             retries=13):
        # "version=2" introduced in dclab 0.54.3
        # (supported since ckanext.dc_serve 0.13.2)
        qstr = f"&version={self.dcserv_api_version}&query={query}"
        if feat is not None:
            qstr += f"&feature={feat}"
        if trace is not None:
            qstr += f"&trace={trace}"
        if event is not None:
            qstr += f"&event={event}"
        apicall = self.url + qstr
        fail_reasons = []
        for _ in range(retries):
            try:
                # try-except both requests and json conversion
                req = self.session.get(apicall,
                                       headers={"Authorization": api_key},
                                       verify=self.verify,
                                       timeout=1,
                                       )
                jreq = req.json()
            except requests.urllib3.exceptions.ConnectionError:  # requests
                fail_reasons.append("connection problem")
                continue
            except (requests.urllib3.exceptions.ReadTimeoutError,
                    requests.exceptions.ConnectTimeout):  # requests
                fail_reasons.append("timeout")
            except json.decoder.JSONDecodeError:  # json
                fail_reasons.append("invalid json")
                time.sleep(1)  # wait a bit, maybe the server is overloaded
                continue
            else:
                break
        else:
            raise DCORAccessError(f"Could not complete query '{apicall}'. "
                                  f"I retried {retries} times. "
                                  f"Messages: {fail_reasons}")
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
