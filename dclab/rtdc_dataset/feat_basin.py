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
import threading
from typing import Dict


class BasinAvailabilityChecker(threading.Thread):
    def __init__(self, basin, *args, **kwargs):
        super(BasinAvailabilityChecker, self).__init__(*args, daemon=True,
                                                       **kwargs)
        self.basin = basin

    def run(self):
        self.basin.is_available()


class Basin(abc.ABC):
    """A basin represents data from an external source

    The external data must be a valid RT-DC dataset, subclasses
    should ensure that the corresponding API is available.
    """
    id_getters = {}

    def __init__(self, location, name=None, description=None,
                 features=None, measurement_identifier=None, **kwargs):
        """

        Parameters
        ----------
        location: str
            Location of the basin, this can be a path or a URL, depending
            on the implementation of the subclass
        name: str
            Human-readable name of the basin
        description: str
            Lengthy descrition of the basin
        features: list of str
            List of features this basin provides; This list is enforced,
            even if the basin actually contains more features.
        measurement_identifier: str
            A measurement identifier against which to check the basin.
            If this is set to None (default), there is no certainty
            that the downstream dataset is from the same measurement.
        """
        #: location of the basin (e.g. path or URL)
        self.location = location
        #: user-defined name of the basin
        self.name = name
        #: lengthy description of the basin
        self.description = description
        # features this basin provides
        self._features = features
        #: measurement identifier of the referencing dataset
        self.measurement_identifier = measurement_identifier
        self._measurement_identifier_verified = False
        #: additional keyword arguments passed to the basin
        self.kwargs = kwargs
        self._ds = None
        # perform availability check in separate thread
        self._av_check_lock = threading.Lock()
        self._av_check = BasinAvailabilityChecker(self)
        self._av_check.start()

    def _assert_measurement_identifier(self):
        """Make sure the basin matches the measurement identifier

        This method caches its result, i.e. only the first call is slow.
        """
        if not self._measurement_identifier_verified:
            if self.measurement_identifier is None:
                self._measurement_identifier_verified = True
            else:
                self._measurement_identifier_verified = (
                    self.get_measurement_identifier()
                    == self.measurement_identifier)
        if not self._measurement_identifier_verified:
            raise KeyError(f"Measurement identifier of {self.ds} "
                           f"({self.get_measurement_identifier()}) does "
                           f"not match {self.measurement_identifier}!")

    @property
    @abc.abstractmethod
    def basin_format(self):
        """Basin format (:class:`.RTDCBase` subclass), e.g. "hdf5" or "s3"
        """
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
        """Features made available by the basin

        .. versionchanged: 0.56.0

           Return nested basin features
        """
        if self._features is None:
            if self.is_available():
                # If features are not specified already, either by previous
                # call to this method or during initialization from basin
                # definition, then make the innate and *all* the basin
                # features available.
                self._features = sorted(set(self.ds.features_innate
                                            + self.ds.features_basin))
            else:
                self._features = []
        return self._features

    def as_dict(self):
        """Return basin kwargs for :func:`RTDCWriter.store_basin`

        Note that each subclass of :class:`.RTDCBase` has its own
        implementation of :func:`.RTDCBase.basins_get_dicts` which
        returns a list of basin dictionaries that are used to
        instantiate the basins in :func:`RTDCBase.basins_enable`.
        This method here is only intended for usage with
        :func:`RTDCWriter.store_basin`.
        """
        return {
            "basin_name": self.name,
            "basin_type": self.basin_type,
            "basin_format": self.basin_format,
            "basin_locs": [self.location],
            "basin_descr": self.description,
            "basin_feats": self.features,
        }

    def close(self):
        """Close any open file handles or connections"""
        if self._ds is not None:
            self._ds.close()
        self._av_check.join(0.5)

    def get_feature_data(self, feat):
        """Return an object representing feature data of the basin"""
        self._assert_measurement_identifier()
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


def basin_priority_sorted_key(bdict: Dict):
    """Yield a sorting value for a given basin that can be used with `sorted`

    Basins are normally stored in random order in a dataset. This method
    brings them into correct order, prioritizing:

    - type "file" over "remote"
    - format "HTTP" over "S3" over "dcor"
    """
    srt_type = {
        "file": "a",
        "remote": "b",
    }.get(bdict.get("type"), "z")

    srt_format = {
        "hdf5": "a",
        "http": "b",
        "s3": "c",
        "dcor": "d",
    }.get(bdict.get("format"), "z")

    return srt_type + srt_format


def get_basin_classes():
    bc = {}
    for bcls in Basin.__subclasses__():
        if hasattr(bcls, "basin_format"):
            bc[bcls.basin_format] = bcls
    return bc
