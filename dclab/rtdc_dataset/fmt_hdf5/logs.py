import functools


class H5Logs:
    def __init__(self, h5):
        self.h5file = h5

    def __getitem__(self, key):
        if key in self.keys():
            log = list(self.h5file["logs"][key])
            if isinstance(log[0], bytes):
                log = [li.decode("utf") for li in log]
        else:
            raise KeyError(
                f"File {self.h5file.file.filename} does not have the log "
                f"'{key}'. Available logs are {self.keys()}.")
        return log

    def __iter__(self):
        # dict-like behavior
        for key in self.keys():
            yield key

    def __len__(self):
        return len(self.keys())

    @functools.lru_cache()
    def keys(self):
        names = []
        if "logs" in self.h5file:
            for key in self.h5file["logs"]:
                if self.h5file["logs"][key].size:
                    names.append(key)
        return names
