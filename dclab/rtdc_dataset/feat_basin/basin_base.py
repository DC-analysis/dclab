from __future__ import annotations

import abc
import logging
import threading
from typing import Literal, TYPE_CHECKING
import uuid
import warnings
import weakref

import numpy as np


from .basin_common import (
    BasinAvailabilityChecker,
    BasinIdentifierMismatchError,
    BasinmapFeatureMissingError,
    BasinNotAvailableError,
)
from .basin_proxy import BasinProxy
from .perishable_record import PerishableRecord

if TYPE_CHECKING:
    from ..core import RTDCBase


logger = logging.getLogger(__name__)


class Basin(abc.ABC):
    """A basin represents data from an external source

    The external data must be a valid RT-DC dataset, subclasses
    should ensure that the corresponding API is available.
    """
    def __init__(self,
                 location: str,
                 name: str | None = None,
                 description: str | None = None,
                 features: list[str] | None = None,
                 referrer_identifier: str | None = None,
                 basin_identifier: str | None = None,
                 mapping: Literal["same",
                                  "basinmap0",
                                  "basinmap1",
                                  "basinmap2",
                                  "basinmap3",
                                  "basinmap4",
                                  "basinmap5",
                                  "basinmap6",
                                  "basinmap7",
                                  "basinmap8",
                                  "basinmap9",
                                  ] = "same",
                 mapping_referrer: dict | RTDCBase | None = None,
                 ignored_basins: list[str] | None = None,
                 key: str | None = None,
                 perishable: bool | PerishableRecord = False,
                 **kwargs):
        """

        Parameters
        ----------
        location: str
            Location of the basin, this can be a path or a URL, depending
            on the implementation of the subclass
        name: str
            Human-readable name of the basin
        description: str
            Lengthy description of the basin
        features: list of str
            List of features this basin provides; This list is enforced,
            even if the basin actually contains more features.
        referrer_identifier: str
            A measurement identifier against which to check the basin.
            If the basin mapping is "same", then this must match the
            identifier of the basin exactly, otherwise it must start
            with the basin identifier (e.g. "basin-id_referrer-sub-id").
            If this is set to None (default), there is no certainty
            that the downstream dataset is from the same measurement.
        basin_identifier: str
            A measurement identifier that must match the basin exactly.
            In contrast to `referrer_identifier`, the basin identifier is
            the identifier of the basin file. If `basin_identifier` is
            specified, the identifier of the basin must be identical to it.
        mapping: str
            Which type of mapping to use. This can be either "same"
            when the event list of the basin is identical to that
            of the dataset defining the basin, or one of the "basinmap"
            features (e.g. "basinmap1") in cases where the dataset consists
            of a subset of the events of the basin dataset. In the latter
            case, the feature defined by `mapping` must be present in the
            dataset and consist of integer-valued indices (starting at 0)
            for the basin dataset.
        mapping_referrer: dict-like
            Dict-like object from which "basinmap" features can be obtained
            in situations where `mapping != "same"`. This can be a simple
            dictionary of numpy arrays or e.g. an instance of
            :class:`.RTDCBase`.
        ignored_basins: list of str
            List of basins to ignore in subsequent basin instantiations
        key: str
            Unique key to identify this basin; normally computed from
            a JSON dump of the basin definition. A random string is used
            if None is specified.
        perishable: bool or PerishableRecord
            If this is not False, then it must be a :class:`.PerishableRecord`
            that holds the information about the expiration time, and that
            comes with a method `refresh` to extend the lifetime of the basin.
        kwargs:
            Additional keyword arguments passed to the `load_dataset`
            method of the `Basin` subclass.

        .. versionchanged: 0.58.0

            Added the `mapping` keyword argument to support basins
            with a superset of events.
        """
        #: location of the basin (e.g. path or URL)
        self.location = location
        #: user-defined name of the basin
        self.name = name
        #: lengthy description of the basin
        self.description = description
        # perishable record
        if isinstance(perishable, bool) and perishable:
            # Create an empty perishable record
            perishable = PerishableRecord(self)
        self.perishable: PerishableRecord | Literal[False] = perishable
        # define key of the basin
        self.key = key or str(uuid.uuid4())
        # features this basin provides
        self._features = features
        #: measurement identifier of the referencing dataset
        self.referrer_identifier = referrer_identifier
        self.basin_identifier = basin_identifier or None
        self._identifiers_verification = None
        #: ignored basins
        self.ignored_basins = ignored_basins or []
        #: additional keyword arguments passed to the basin
        self.kwargs = kwargs
        #: Event mapping strategy. If this is "same", it means that the
        #: referring dataset and the basin dataset have identical event
        #: indices. If `mapping` is e.g. `basinmap1` then the mapping of the
        #: indices from the basin to the referring dataset is defined in
        #: `self.basinmap` (copied during initialization of this class from
        #: the array in the key `basinmap1` from the dict-like object
        #: `mapping_referrer`).
        self.mapping = mapping or "same"
        self._basinmap = None  # see `basinmap` property
        # Create a weakref to the original referrer: If it is an instance
        # of RTDCBase, then garbage collection can clean up properly and
        # the basin instance has no reason to exist without the referrer.
        if mapping_referrer is not None:
            self._basinmap_referrer = weakref.ref(mapping_referrer)
        else:
            self._basinmap_referrer = None
        self._ds: RTDCBase | BasinProxy | None = None
        # perform availability check in separate thread
        self._av_check_lock = threading.Lock()
        self._av_check = BasinAvailabilityChecker(self)
        self._av_check.start()

    def __repr__(self):
        try:
            feature_info = len(self.features)
        except BaseException:
            feature_info = "unknown"
        options = [
            self.name,
            f"mapped {self.mapping}" if self.mapping != "same" else "",
            f"{feature_info} features",
            f"location {self.location}",
        ]
        opt_str = ", ".join([o for o in options if o])

        return f"<{self.__class__.__name__} ({opt_str}) at {hex(id(self))}>"

    def _assert_referrer_identifier(self):
        """Make sure the basin matches the measurement identifier
        """
        if not self.verify_basin(run_identifier=True):
            raise BasinIdentifierMismatchError(
                f"Measurement identifier of basin {self.ds} "
                f"({self.get_measurement_identifier()}) does "
                f"not match {self.referrer_identifier}!")

    @property
    def basinmap(self):
        """Contains the indexing array in case of a mapped basin"""
        if self._basinmap is None:
            if self.mapping != "same":
                assert self._basinmap_referrer is not None
                ref = self._basinmap_referrer()
                assert ref is not None
                try:
                    basinmap = ref[self.mapping]
                except (KeyError, RecursionError):
                    raise BasinmapFeatureMissingError(
                        f"Could not find the feature '{self.mapping}' in the "
                        f"dataset or any of its basins. This suggests that "
                        f"this feature was never saved anywhere. Please check "
                        f"the input files.")
                #: `basinmap` is an integer array that maps the events from the
                #: basin to the events of the referring dataset.
                self._basinmap = np.array(basinmap,
                                          dtype=np.uint64,
                                          copy=True)
            else:
                self._basinmap = None
        return self._basinmap

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
        if self.perishable and self.perishable.perished():
            # We have perished. Ask the PerishableRecord to refresh this
            # basin so we can access it again.
            self.perishable.refresh()
        if self._ds is None:
            if not self.is_available():
                raise BasinNotAvailableError(f"Basin {self} is not available!")
            self._ds = self.load_dataset(self.location, **self.kwargs)
            assert self._ds is not None
            self._ds.ignore_basins(self.ignored_basins)
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
            "basin_map": self.basinmap,
            "perishable": bool(self.perishable),
        }

    def close(self):
        """Close any open file handles or connections"""
        if self._ds is not None:
            self._ds.close()
        self._av_check.join(0.5)

    def get_feature_data(self, feat):
        """Return an object representing feature data of the basin"""
        self._assert_referrer_identifier()
        return self.ds[feat]

    def get_measurement_identifier(self):
        """Return the identifier of the basin dataset"""
        return self.ds.get_measurement_identifier()

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if the basin is available"""

    @abc.abstractmethod
    def _load_dataset(self, location, **kwargs) -> RTDCBase:
        """Subclasses should return an instance of :class:`.RTDCBase`"""

    def load_dataset(self, location, **kwargs):
        """Return an instance of :class:`.RTDCBase` for this basin

        If the basin mapping (`self.mapping`) is not the same as the
        referencing dataset (`self.mapping != "same"`), return a
        `BasinProxy` object.
        """
        ds = self._load_dataset(location, **kwargs)
        if self.mapping != "same":
            # The array `self.basinmap` may contain duplicate elements,
            # which is why we cannot use hierarchy children to access the
            # data (sometimes the data must be blown-up rather than gated).
            ds_bn = BasinProxy(ds=ds, basinmap=self.basinmap)
        else:
            ds_bn = ds
        return ds_bn

    def verify_basin(self, run_identifier=True, availability=True):
        if not availability:
            warnings.warn("The keyword argument 'availability' is "
                          "deprecated, because it can lead to long waiting "
                          "times with many unavailable basins.",
                          DeprecationWarning)
        if availability:
            check_avail = self.is_available()
        else:
            check_avail = True

        # Only check for run identifier if requested and if the availability
        # check did not fail.
        if run_identifier and check_avail:
            if self._identifiers_verification is None:
                # This is the measurement identifier of the basin.
                basin_identifier = self.get_measurement_identifier()

                # Perform a sanity check for the basin identifier.
                if (self.basin_identifier
                        and self.basin_identifier != basin_identifier):
                    # We should not proceed any further with this basin.
                    self._identifiers_verification = False
                    warnings.warn(
                        f"Basin identifier mismatch for {self}. Expected "
                        f"'{self.basin_identifier}', got '{basin_identifier}'")

                if self.referrer_identifier is None:
                    # No measurement identifier was presented by the
                    # referencing dataset. We are in the dark.
                    # Don't perform any checks.
                    self._identifiers_verification = True
                else:
                    if basin_identifier is None:
                        # Again, we are in the dark, because the basin dataset
                        # does not have an identifier. This is an undesirable
                        # situation, but there is nothing we can do about it.
                        self._identifiers_verification = True
                    else:
                        if self.mapping == "same":
                            # When we have identical mapping, then the
                            # measurement identifier has to match exactly.
                            verifier = str.__eq__
                        else:
                            # When we have non-identical mapping (e.g. exported
                            # data), then the measurement identifier has to
                            # partially match.
                            verifier = str.startswith
                        self._identifiers_verification = verifier(
                            self.referrer_identifier, basin_identifier)

            check_rid = self._identifiers_verification
        else:
            check_rid = True

        return check_rid and check_avail


def get_basin_classes():
    bc = {}
    for b_cls in Basin.__subclasses__():
        if hasattr(b_cls, "basin_format"):
            bc[b_cls.basin_format] = b_cls
    return bc
