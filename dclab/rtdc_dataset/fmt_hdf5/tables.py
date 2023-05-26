import functools


class H5Tables:
    def __init__(self, h5):
        self.h5file = h5

    def __getitem__(self, key):
        if key in self.keys():
            tab = self.h5file["tables"][key]
        else:
            raise KeyError(f"Table '{key}' not found or empty "
                           f"in {self.h5file.file.filename}!")
        return tab

    def __iter__(self):
        # dict-like behavior
        for key in self.keys():
            yield key

    def __len__(self):
        return len(self.keys())

    @functools.lru_cache()
    def keys(self):
        names = []
        if "tables" in self.h5file:
            for key in self.h5file["tables"]:
                if self.h5file["tables"][key].size:
                    names.append(key)
        return names
