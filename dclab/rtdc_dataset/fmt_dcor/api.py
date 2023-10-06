import json
import time

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
    #: DCOR API Keys/Tokens in the current session
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
        #: DCOR API Token
        self.api_key = api_key
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
