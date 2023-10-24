import functools


class DCORLogs:
    def __init__(self, api):
        self.api = api

    def __contains__(self, key):
        return key in self.keys()

    def __getitem__(self, key):
        return self._logs[key]

    def __len__(self):
        return len(self._logs)

    def keys(self):
        return self._logs.keys()

    @property
    @functools.lru_cache()
    def _logs(self):
        return self.api.get(query="logs")
