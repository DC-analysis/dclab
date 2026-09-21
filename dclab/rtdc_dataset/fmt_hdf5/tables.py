from __future__ import annotations

from collections.abc import Iterator

import h5py
import numpy as np

from ..meta_table import MetaTable


class H5Tables:
    def __init__(self, h5: h5py.Group) -> None:
        self.h5file = h5
        self._cache_keys: list[str] | None = None

    def __getitem__(self, key: str) -> H5Table:
        if key in self.keys():
            tab = H5Table(self.h5file["tables"][key])  # type: ignore
        else:
            raise KeyError(f"Table '{key}' not found or empty "
                           f"in {self.h5file.file.filename}!")
        return tab

    def __iter__(self) -> Iterator[str]:
        # dict-like behavior
        yield from self.keys()

    def __len__(self) -> int:
        return len(self.keys())

    def keys(self) -> list[str]:
        if self._cache_keys is None:
            names = []
            if "tables" in self.h5file:
                for key in self.h5file["tables"]:  # type: ignore
                    if self.h5file["tables"][key].size:  # type: ignore
                        names.append(key)
            self._cache_keys = names
        return self._cache_keys


class H5Table(MetaTable):
    def __init__(self, h5_ds: h5py.Dataset) -> None:
        self._h5_ds = h5_ds
        self._keys: tuple[str, ...] | None = None
        self._meta: dict | None = None

    def __array__(self, *args, **kwargs) -> np.ndarray:
        return self._h5_ds.__array__(*args, **kwargs)

    @property
    def meta(self) -> dict:
        if self._meta is None:
            self._meta = dict(self._h5_ds.attrs)
        return self._meta

    def has_graphs(self) -> bool:
        return self.keys() is not None

    def keys(self) -> tuple[str, ...] | None:  # type: ignore
        if self._keys is None:
            self._keys = self._h5_ds.dtype.names
        return self._keys

    def __getitem__(self, key: str) -> np.ndarray:
        return self._h5_ds[key]
