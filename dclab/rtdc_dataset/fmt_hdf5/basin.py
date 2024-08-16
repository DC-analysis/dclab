"""RT-DC hdf5 format"""
from __future__ import annotations

import pathlib

from .. import feat_basin

from .base import RTDC_HDF5


class HDF5Basin(feat_basin.Basin):
    basin_format = "hdf5"
    basin_type = "file"

    def __init__(self, *args, **kwargs):
        self._available_verified = None
        super(HDF5Basin, self).__init__(*args, **kwargs)

    def _load_dataset(self, location, **kwargs):
        return RTDC_HDF5(location, **kwargs)

    def is_available(self):
        if self._available_verified is None:
            with self._av_check_lock:
                try:
                    self._available_verified = \
                        pathlib.Path(self.location).exists()
                except OSError:
                    pass
        return self._available_verified
