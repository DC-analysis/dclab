class DCORLogs:
    def __init__(self, api):
        self.api = api
        self._logs_cache = None

    def __contains__(self, key):
        return key in self.keys()

    def __getitem__(self, key):
        return self._logs[key]

    def __len__(self):
        return len(self._logs)

    def keys(self):
        return self._logs.keys()

    @property
    def _logs(self):
        if self._logs_cache is None:
            self._logs_cache = self.api.get(query="logs")
        return self._logs_cache
