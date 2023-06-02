"""
Basins are other .rtdc files on disk or online (DCOR, S3) which originate
from the same dataset but (potentially) contain more features. Basins
are useful if you would like to have a small copy of selected
features in a separate file while still being able to access
all features from the original file. E.g. you could load a small
.rtdc file from your local disk and access the larger "image"
feature data from an S3 basin. Basins are active by default, which
means that if you load a dataset that defines basins and these
basins are available, they will be integrated seamlessly into your
analysis pipeline. You can find out which features originate from
other basins via the ``features_basin`` property of an
:class:`.RTDCBase` instance.
"""
from __future__ import annotations

import abc


class Basin(abc.ABC):
    """A basin represents data from an external source

    The external data must be a valid RT-DC dataset, subclasses
    should ensure that the corresponding API is available.
    """
    id_getters = {}

    def __init__(self, location, **kwargs):
        self.location = location
        self.kwargs = kwargs
        self._ds = None

    @property
    @abc.abstractmethod
    def basin_format(self):
        """Basin format (:class:`.RTDCBase` subclass)"""
        # to be implemented in subclasses

    @property
    @abc.abstractmethod
    def basin_type(self):
        """Storage type to use (e.g. "file" or "remote")"""
        # to be implemented in subclasses

    @property
    def ds(self):
        """The :class:`.RTDCBase` instance represented by the basin"""
        if self._ds is None:
            if not self.is_available():
                raise ValueError(f"Basin {self} is not available!")
            self._ds = self.load_dataset(self.location, **self.kwargs)
        return self._ds

    @property
    def features(self):
        """Features made available by the basin"""
        return self.ds.features_innate

    def get_feature_data(self, feat):
        """Return an object representing feature data of the basin"""
        return self.ds[feat]

    def get_measurement_identifier(self):
        """Return the identifier of the dataset"""
        return self.ds.get_measurement_identifier()

    @abc.abstractmethod
    def is_available(self):
        """Return True if the basin is available"""

    @abc.abstractmethod
    def load_dataset(self, location, **kwargs):
        """Subclasses should return an instance of :class:`.RTDCBase`"""


def get_basin_classes():
    bc = {}
    for bcls in Basin.__subclasses__():
        if hasattr(bcls, "basin_format"):
            bc[bcls.basin_format] = bcls
    return bc
