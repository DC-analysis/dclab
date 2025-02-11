from ..meta_table import MetaTable


class H5Tables:
    def __init__(self, h5):
        self.h5file = h5
        self._cache_keys = None

    def __getitem__(self, key):
        if key in self.keys():
            tab = H5Table(self.h5file["tables"][key])
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

    def keys(self):
        if self._cache_keys is None:
            names = []
            if "tables" in self.h5file:
                for key in self.h5file["tables"]:
                    if self.h5file["tables"][key].size:
                        names.append(key)
            self._cache_keys = names
        return self._cache_keys


class H5Table(MetaTable):
    def __init__(self, h5_ds):
        self._h5_ds = h5_ds
        self._keys = None
        self._meta = None

    def __array__(self, *args, **kwargs):
        return self._h5_ds.__array__(*args, **kwargs)

    @property
    def meta(self):
        if self._meta is None:
            self._meta = dict(self._h5_ds.attrs)
        return self._meta

    def has_graphs(self):
        return self.keys() is not None

    def keys(self):
        if self._keys is None:
            self._keys = self._h5_ds.dtype.names
        return self._keys

    def __getitem__(self, key):
        return self._h5_ds[key]
