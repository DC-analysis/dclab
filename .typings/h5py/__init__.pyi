# Local type stub for h5py, used by pyright (see `[tool.pyright]` in
# pyproject.toml). h5py does not ship type hints, and pyright infers
# unions from its source code (e.g. `Group.get` may return a `HardLink`),
# which results in many false positives. This stub makes everything
# in h5py `Any`. Remove it once h5py ships type hints:
# https://github.com/h5py/h5py/issues/1912
from typing import Any

def __getattr__(name: str) -> Any: ...
