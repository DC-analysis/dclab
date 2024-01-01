"""RT-DC hdf5 format"""
from __future__ import annotations

import pathlib

from .. import feat_basin

from .base import RTDC_HDF5


class HDF5Basin(feat_basin.Basin):
    basin_format = "hdf5"
    basin_type = "file"

    def load_dataset(self, location, **kwargs):
        return RTDC_HDF5(location, enable_basins=False, **kwargs)

    def is_available(self):
        with self._av_check_lock:
            avail = False
            try:
                avail = pathlib.Path(self.location).exists()
            except OSError:
                pass
        return avail
