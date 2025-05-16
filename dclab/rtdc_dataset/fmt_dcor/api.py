import json
import time

from ...http_utils import REQUESTS_AVAILABLE  # noqa: F401
from ...http_utils import requests, session_cache


class DCORAccessError(BaseException):
    pass


class APIHandler:
    """Handles the DCOR api with caching for simple queries"""
    #: These are cached to minimize network usage
    #: Note that we are not caching basins, since they may contain
    #: expiring URLs.
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

    def _get(self,
             query: str,
             feat: str = None,
             trace: str = None,
             event: str = None,
             api_key: str = "",
             timeout: float = None,
             retries: int = 5):
        """Fetch information via the DCOR API

        Parameters
        ----------
        query: str
            API route
        feat: str
            DEPRECATED (use basins instead), adds f"&feature={feat}" to query
        trace: str
            DEPRECATED (use basins instead), adds f"&trace={trace}" to query
        event: str
            DEPRECATED (use basins instead), adds f"&event={event}" to query
        api_key: str
            DCOR API token to use
        timeout: float
            Request timeout
        retries: int
            Number of retries to fetch the request. For every retry, the
            timeout is increased by two seconds.
        """
        if timeout is None:
            timeout = 1
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
        for ii in range(retries):
            try:
                # try-except both requests and json conversion
                req = self.session.get(apicall,
                                       headers={"Authorization": api_key},
                                       verify=self.verify,
                                       timeout=timeout + ii * 2,
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

    def get(self,
            query: str,
            feat: str = None,
            trace: str = None,
            event: str = None,
            timeout: float = None,
            retries: int = 5,
            ):
        """Fetch information from DCOR

        Parameters
        ----------
        query: str
            API route
        feat: str
            DEPRECATED (use basins instead), adds f"&feature={feat}" to query
        trace: str
            DEPRECATED (use basins instead), adds f"&trace={trace}" to query
        event: str
            DEPRECATED (use basins instead), adds f"&event={event}" to query
        timeout: float
            Request timeout
        retries: int
            Number of retries to fetch the request. For every retry, the
            timeout is increased by two seconds.
        """
        if query in APIHandler.cache_queries and query in self._cache:
            result = self._cache[query]
        else:
            req = {"error": {"message": "No access to API (api key?)"}}
            for api_key in [self.api_key] + APIHandler.api_keys:
                req = self._get(query=query,
                                feat=feat,
                                trace=trace,
                                event=event,
                                api_key=api_key,
                                timeout=timeout,
                                retries=retries,
                                )
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
