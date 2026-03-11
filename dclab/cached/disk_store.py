import json
import numbers
import pathlib
import shutil
import time

import numpy as np


class DiskStore:
    version = "1.0"

    def __init__(self, path=None):
        """Disk store for persisting data to disk"""
        self.path = None
        self.index = set()
        self.size_bytes = 0
        if path is not None:
            self.set_path(path)

    def __bool__(self):
        return self.path is not None

    def __contains__(self, key):
        if self:
            return (key in self.index
                    or (self.path / f"{key}_meta.json").is_file())
        else:
            return False

    def __getitem__(self, key):
        self.assert_disk_store_path()
        meta_path = self.path / f"{key}_meta.json"
        file_meta = json.loads(meta_path.read_text())
        meta_path.touch()
        return self.value_read(key, file_meta)

    def __setitem__(self, key, value):
        file_meta = self.value_write(key, value)
        meta_path = self.path / f"{key}_meta.json"
        file_meta["version"] = self.version
        file_meta["timestamp"] = time.time()
        self.index.add(key)
        meta_path.write_text(json.dumps(file_meta, indent=2))

    def assert_disk_store_path(self):
        if not self:
            raise RuntimeError(
                "Please use `set_disk_store_path` to set a disk cache path")

    def clear(self):
        if self:
            self.index.clear()
            shutil.rmtree(self.path, ignore_errors=True)

    def remove_old_files(self, max_bytes):
        self.assert_disk_store_path()
        # get the sizes and times of all cache items
        items = []
        keys = []
        total_bytes = 0
        for pp in self.path.rglob("*_meta.json"):
            meta = json.loads(pp.read_text())
            stem = pp.with_name(pp.name.split("_")[0])
            key = str(stem.relative_to(self.path))
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
                for pi in self.path.glob(f"{key}*"):
                    if pi.exists():
                        pi.unlink()
                if total_bytes < max_bytes:
                    break

    def set_path(self, path):
        path = pathlib.Path(path)
        if path != self.path:
            self.path = pathlib.Path(path)
            # trigger an index update
            self.index.clear()

    def value_read(self, key, file_meta):
        if file_meta["format"] == "json":
            return json.loads(
                ((self.path / key).parent / file_meta["file"]).read_text())
        elif file_meta["format"] == "numpy":
            return np.load((self.path / key).parent / file_meta["file"])
        elif file_meta["format"] == "list":
            data = []
            for fmi in file_meta["items"]:
                data.append(self.value_read(key, fmi))
            return data
        else:
            raise NotImplementedError(
                f"Unsupported format '{file_meta['format']}'")

    def value_write(self, key, value):
        (self.path / key).parent.mkdir(parents=True, exist_ok=True)
        # first attempt to just store everything as json
        try:
            json_data = json.dumps(value, cls=DiskStoreJSONEncoder, indent=2)
        except BaseException:
            json_data = None

        if json_data is not None:
            return self.value_write_json(key, json_data)
        else:
            return self.value_write_mixed(key, value)

    def value_write_json(self, key, json_data):
        path_out = pathlib.Path(self.path / f"{key}.json")
        path_out.write_text(json_data)
        file_meta = {"format": "json",
                     "file": path_out.name,
                     "size": path_out.stat().st_size
                     }
        return file_meta

    def value_write_mixed(self, key, value):
        if isinstance(value, (list, tuple)):
            fmis = []
            for ii in range(len(value)):
                fmi = self.value_write(f"{key}_{ii}", value[ii])
                fmis.append(fmi)
            file_meta = {"format": "list",
                         "items": fmis,
                         "size": sum([f["size"] for f in fmis]),
                         }
        elif isinstance(value, np.ndarray):
            path_out = self.path / f"{key}.npy"
            np.save(path_out, value)
            file_meta = {"format": "numpy",
                         "file": path_out.name,
                         "size": path_out.stat().st_size
                         }
        else:
            raise NotImplementedError(
                f"No recipe to store '{key}' of type {type(value)}")
        return file_meta


class DiskStoreJSONEncoder(json.JSONEncoder):
    """Custom JSONEncoder"""
    def default(self, obj):
        if isinstance(obj, pathlib.Path):
            return str(obj)
        elif isinstance(obj, numbers.Integral):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
