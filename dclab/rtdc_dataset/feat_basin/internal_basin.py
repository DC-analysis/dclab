from __future__ import annotations

import warnings

from .basin_base import Basin
from .basin_common import BasinFeatureMissingWarning


class InternalH5DatasetBasin(Basin):
    basin_format = "h5dataset"
    basin_type = "internal"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mapping == "same":
            raise ValueError(
                "'internal' basins must be instantiated with `mapping`. "
                "If you are not doing that, then you probably don't need "
                "them.")
        if self._features is None:
            raise ValueError("You must specify features when defining "
                             "internal basins.")
        # Redefine the features if necessary
        h5root = self._get_h5file()
        available_features = []
        for feat in self._features:
            if self.location in h5root and feat in h5root[self.location]:
                available_features.append(feat)
            else:
                warnings.warn(
                    f"Feature '{feat}' is defined as an internal basin, "
                    f"but it cannot be found in '{self.location}'.",
                    BasinFeatureMissingWarning)
        self._features.clear()
        self._features += available_features

    def _get_h5file(self):
        assert self._basinmap_referrer is not None
        ref = self._basinmap_referrer()
        assert ref is not None
        return ref.h5file

    def _load_dataset(self, location, **kwargs):
        # to avoid circular imports...
        from ..fmt_dict import RTDC_Dict
        # get the h5file object
        h5root = self._get_h5file()
        # fetch data
        ds_dict = {}
        for feat in self.features:
            ds_dict[feat] = h5root[self.location][feat]
        return RTDC_Dict(ds_dict)

    def is_available(self):
        return bool(self._features)

    def verify_basin(self, *args, **kwargs):
        """It's not necessary to verify internal basins"""
        return True
