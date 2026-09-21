from __future__ import annotations

from collections.abc import Iterator

import h5py


class H5Logs:
    def __init__(self, h5: h5py.Group) -> None:
        self.h5file = h5
        self._cache_keys: list[str] | None = None

    def __getitem__(self, key: str) -> list[str]:
        if key in self.keys():
            log = list(self.h5file["logs"][key])  # type: ignore
            if isinstance(log[0], bytes):
                log = [li.decode("utf") for li in log]
        else:
            raise KeyError(
                f"File {self.h5file.file.filename} does not have the log "
                f"'{key}'. Available logs are {self.keys()}.")
        return log

    def __iter__(self) -> Iterator[str]:
        # dict-like behavior
        yield from self.keys()

    def __len__(self) -> int:
        return len(self.keys())

    def keys(self) -> list[str]:
        if self._cache_keys is None:
            names = []
            if "logs" in self.h5file:
                for key in self.h5file["logs"]:  # type: ignore
                    if self.h5file["logs"][key].size:  # type: ignore
                        names.append(key)
            self._cache_keys = names
        return self._cache_keys
