"""Persistent on-disk store for :class:`.umbrella_cache`"""
import json
import numbers
import os
import pathlib
import shutil
import time

import numpy as np


class DiskStore:
    #: version of the disk store file format
    version = "1.0"

    def __init__(self, path=None):
        """Disk store for persisting data to disk

        The disk store is thread- and process-safe.

        Parameters
        ----------
        path: str or pathlib.Path
            Directory where the data are stored; if None, the disk
            store is disabled until :func:`set_path` is called
        """
        #: directory of the disk store (None if the disk store is disabled)
        self.path: pathlib.Path | None = None
        self.index = set()
        self.size_bytes = 0
        if path is not None:
            self.set_path(path)

    def __bool__(self):
        return self.path is not None

    def __contains__(self, key):
        if self:
            return (key in self.index
                    or (self._store_path / f"{key}_meta.json").is_file())
        else:
            return False

    def __getitem__(self, key):
        self.assert_disk_store_path()
        meta_path = self._store_path / f"{key}_meta.json"
        file_meta = json.loads(meta_path.read_text())
        meta_path.touch()
        return self.value_read(key, file_meta)

    def __setitem__(self, key, value):
        lock_path = self._store_path / f"{key}.lock"
        # Use a lock file to avoid race conditions with other
        # processes when writing data to disk.
        # If the lock cannot be acquired, then nothing is written
        # to disk. We trust that the other disk store is storing
        # the correct data to disk.
        with LockFile(lock_path) as lf:
            if lf.acquired:
                file_meta = self.value_write(key, value)
                meta_path = self._store_path / f"{key}_meta.json"
                file_meta["version"] = self.version
                file_meta["timestamp"] = time.time()
                self.index.add(key)
                meta_path.write_text(json.dumps(file_meta, indent=2))

    @property
    def _store_path(self) -> pathlib.Path:
        """Directory of the disk store (RuntimeError if not set)"""
        if self.path is None:
            raise RuntimeError(
                "Please use `set_disk_store_path` to set a disk cache path")
        return self.path

    def assert_disk_store_path(self):
        """Raise a RuntimeError if no disk store path is set"""
        self._store_path  # noqa: B018

    def clear(self):
        """Remove all data from the disk store (deletes the store directory)"""
        if self:
            self.index.clear()
            shutil.rmtree(self._store_path, ignore_errors=True)

    def remove(self, key):
        """Remove all files of `key` from the disk store"""
        self.assert_disk_store_path()
        self.index.discard(key)
        # Remove the meta file first, so that `key` is not considered
        # to be in the disk store anymore.
        (self._store_path / f"{key}_meta.json").unlink(missing_ok=True)
        for pi in self._store_path.glob(f"{key}*"):
            if pi.suffix != ".lock":
                pi.unlink(missing_ok=True)

    def remove_old_files(self, max_bytes):
        """Remove old files, honoring `max_bytes` total size"""
        self.assert_disk_store_path()
        # get the sizes and times of all cache items
        items = []
        keys = []
        total_bytes = 0
        for pp in self._store_path.rglob("*_meta.json"):
            meta = json.loads(pp.read_text())
            stem = pp.with_name(pp.name.split("_")[0])
            key_path = str(stem.relative_to(self._store_path))
            # On Windows, paths are separated with "\", but our keys
            # are labeled with "/".
            key = "/".join(key_path.split(os.path.sep))
            keys.append(key)
            items.append([pp.stat().st_mtime, meta["size"], key])
            total_bytes += meta["size"]
        self.index.clear()
        self.index.update(keys)

        items = sorted(items)

        if total_bytes > max_bytes:
            for (_, size, key) in items:
                total_bytes -= size
                try:
                    self.index.remove(key)
                except KeyError:
                    pass
                for pi in self._store_path.glob(f"{key}*"):
                    if pi.exists():
                        pi.unlink()
                if total_bytes < max_bytes:
                    break

    def remove_stale_locks(self, max_age_seconds=3600):
        """Remove stale locks"""
        for pp in self._store_path.rglob("*.lock"):
            if pp.stat().st_mtime < time.time() - max_age_seconds:
                pp.unlink(missing_ok=True)

    def set_path(self, path):
        """Set the directory in which the disk store keeps its data"""
        path = pathlib.Path(path)
        if path != self.path:
            self.path = pathlib.Path(path)
            # trigger an index update
            self.index.clear()

    def value_read(self, key, file_meta):
        """Read the value of `key` as described by the metadata `file_meta`"""
        if file_meta["format"] == "list":
            data = []
            for fmi in file_meta["items"]:
                data.append(self.value_read(key, fmi))
            if file_meta.get("type") == "tuple":
                data = tuple(data)
            return data

        path_data = (self._store_path / key).parent / file_meta.get("file", "")
        if file_meta["format"] == "json":
            return json.loads(path_data.read_text())
        elif file_meta["format"] == "json-typed":
            return json.loads(path_data.read_text(),
                              object_hook=json_decode_types)
        elif file_meta["format"] == "numpy":
            return np.load(path_data)
        elif file_meta["format"] == "bytes":
            return path_data.read_bytes()
        else:
            raise NotImplementedError(
                f"Unsupported format '{file_meta['format']}'")

    def value_write(self, key, value):
        """Write `value` for `key` to disk and return its metadata

        The value is stored as JSON if possible, otherwise via
        :func:`value_write_mixed`.
        """
        (self._store_path / key).parent.mkdir(parents=True, exist_ok=True)
        # first attempt to just store everything as json
        # (tuples and non-string dictionary keys require type information)
        typed = has_json_lossy_types(value)
        try:
            json_data = json.dumps(
                json_encode_types(value) if typed else value,
                cls=DiskStoreJSONEncoder, indent=2)
        except BaseException:
            json_data = None

        if json_data is not None:
            file_meta = self.value_write_json(key, json_data)
            if typed:
                file_meta["format"] = "json-typed"
            return file_meta
        else:
            return self.value_write_mixed(key, value)

    def value_write_json(self, key, json_data):
        """Write the JSON string `json_data` for `key`, return its metadata"""
        path_out = pathlib.Path(self._store_path / f"{key}.json")
        path_out_tmp = path_out.with_name(path_out.name + "~")
        if path_out_tmp.exists():
            path_out_tmp.unlink()
        path_out_tmp.write_text(json_data)
        if path_out.exists():
            path_out.unlink()
        path_out_tmp.rename(path_out)
        file_meta = {"format": "json",
                     "file": path_out.name,
                     "size": path_out.stat().st_size
                     }
        return file_meta

    def value_write_mixed(self, key, value):
        """Write a list, tuple, numpy array, or bytes value for `key`

        Returns the metadata of the stored value. Raises a
        NotImplementedError for other types.
        """
        if isinstance(value, (list, tuple)):
            fmis = []
            for ii in range(len(value)):
                fmi = self.value_write(f"{key}_{ii}", value[ii])
                fmis.append(fmi)
            file_meta = {"format": "list",
                         "items": fmis,
                         "size": sum([f["size"] for f in fmis]),
                         }
            if isinstance(value, tuple):
                file_meta["type"] = "tuple"
        elif isinstance(value, np.ndarray):
            path_out = self._store_path / f"{key}.npy"
            path_out_tmp = path_out.with_name(path_out.name + "~")
            if path_out_tmp.exists():
                path_out_tmp.unlink()
            with path_out_tmp.open("wb") as fd:
                np.save(fd, value)
            if path_out.exists():
                path_out.unlink()
            path_out_tmp.rename(path_out)

            file_meta = {"format": "numpy",
                         "file": path_out.name,
                         "size": path_out.stat().st_size
                         }
        elif isinstance(value, bytes):
            path_out = self._store_path / f"{key}.bin"
            path_out_tmp = path_out.with_name(path_out.name + "~")
            if path_out_tmp.exists():
                path_out_tmp.unlink()
            path_out_tmp.write_bytes(value)
            if path_out.exists():
                path_out.unlink()
            path_out_tmp.rename(path_out)

            file_meta = {"format": "bytes",
                         "file": path_out.name,
                         "size": path_out.stat().st_size
                         }
        else:
            raise NotImplementedError(
                f"No recipe to store '{key}' of type {type(value)}")
        return file_meta


class DiskStoreJSONEncoder(json.JSONEncoder):
    """Custom JSONEncoder"""
    def default(self, o):
        """Convert paths, integers, and ``numpy.bool_`` objects for JSON"""
        if isinstance(o, pathlib.Path):
            return str(o)
        elif isinstance(o, numbers.Integral):
            return int(o)
        elif isinstance(o, np.bool_):
            return bool(o)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)


def has_json_lossy_types(value):
    """Whether `value` contains objects that JSON cannot represent exactly

    JSON does not distinguish between tuples and lists and only
    supports strings as dictionary keys.
    """
    if isinstance(value, tuple):
        return True
    elif isinstance(value, list):
        return any(has_json_lossy_types(v) for v in value)
    elif isinstance(value, dict):
        return (any(not isinstance(k, str) for k in value)
                or any(has_json_lossy_types(v) for v in value.values()))
    return False


def json_encode_types(value):
    """Encode tuples and dictionaries for JSON, see `json_decode_types`"""
    if isinstance(value, tuple):
        return {"__dclab_tuple__": [json_encode_types(v) for v in value]}
    elif isinstance(value, list):
        return [json_encode_types(v) for v in value]
    elif isinstance(value, dict):
        return {"__dclab_dict__": [[json_encode_types(k), json_encode_types(v)]
                                   for k, v in value.items()]}
    return value


def json_decode_types(obj):
    """Object hook for `json.loads` to decode `json_encode_types` data"""
    if len(obj) == 1:
        if "__dclab_tuple__" in obj:
            return tuple(obj["__dclab_tuple__"])
        elif "__dclab_dict__" in obj:
            return {k: v for k, v in obj["__dclab_dict__"]}
    return obj


class LockFile:
    def __init__(self, path):
        """Create a lock file at the specified path

        Use the attribute `acquired` to check whether the
        lock has been acquired.
        """
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("x") as fd:
                fd.write(f"{os.getpid()}")  # for debugging
        except FileExistsError:
            self.acquired = False
        else:
            self.acquired = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            self.path.unlink(missing_ok=True)
