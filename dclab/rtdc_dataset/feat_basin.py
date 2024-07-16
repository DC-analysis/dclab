"""
With basins, you can create analysis pipelines that result in output files
which, when opened in dclab, can access features stored in the input file
(without having to write those features to the output file).
"""
from __future__ import annotations

import abc
import numbers
import threading
from typing import Dict, List, Literal
import warnings
import weakref

import numpy as np


class CyclicBasinDependencyFoundWarning(UserWarning):
    """Used when a basin is defined in one of its sub-basins"""


class BasinmapFeatureMissingError(KeyError):
    """Used when one of the `basinmap` features is not defined"""
    pass


class BasinNotAvailableError(BaseException):
    """Used to identify situations where the basin data is not available"""
    pass


class BasinAvailabilityChecker(threading.Thread):
    """Helper thread for checking basin availability in the background"""
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

    def __init__(self,
                 location: str,
                 name: str = None,
                 description: str = None,
                 features: List[str] = None,
                 measurement_identifier: str = None,
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
                 mapping_referrer: Dict = None,
                 ignored_basins: List[str] = None,
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
        measurement_identifier: str
            A measurement identifier against which to check the basin.
            If this is set to None (default), there is no certainty
            that the downstream dataset is from the same measurement.
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
        # features this basin provides
        self._features = features
        #: measurement identifier of the referencing dataset
        self.measurement_identifier = measurement_identifier
        self._measurement_identifier_verified = False
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
        if self.mapping != "same":
            self._basinmap_referrer = weakref.ref(mapping_referrer)
        else:
            self._basinmap_referrer = None
        self._ds = None
        # perform availability check in separate thread
        self._av_check_lock = threading.Lock()
        self._av_check = BasinAvailabilityChecker(self)
        self._av_check.start()

    def __repr__(self):
        options = [
            self.name,
            f"mapped {self.mapping}" if self.mapping != "same" else "",
            f"features {self._features}" if self.features else "full-featured",
            f"location {self.location}",
        ]
        opt_str = ", ".join([o for o in options if o])

        return f"<{self.__class__.__name__} ({opt_str}) at {hex(id(self))}>"

    def _assert_measurement_identifier(self):
        """Make sure the basin matches the measurement identifier
        """
        if not self.verify_basin(run_identifier=True):
            raise KeyError(f"Measurement identifier of basin {self.ds} "
                           f"({self.get_measurement_identifier()}) does "
                           f"not match {self.measurement_identifier}!")

    @property
    def basinmap(self):
        """Contains the indexing array in case of a mapped basin"""
        if self._basinmap is None:
            if self.mapping != "same":
                try:
                    basinmap = self._basinmap_referrer()[self.mapping]
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
        if self._ds is None:
            if not self.is_available():
                raise BasinNotAvailableError(f"Basin {self} is not available!")
            self._ds = self.load_dataset(self.location, **self.kwargs)
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
        """Return the identifier of the basin dataset"""
        return self.ds.get_measurement_identifier()

    @abc.abstractmethod
    def is_available(self):
        """Return True if the basin is available"""

    @abc.abstractmethod
    def _load_dataset(self, location, **kwargs):
        """Subclasses should return an instance of :class:`.RTDCBase`"""

    def load_dataset(self, location, **kwargs):
        """Return an instance of :class:`.RTDCBase` for this basin

        If the basin mapping (`self.mapping`) is not the same as
        the referencing dataset
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
            if not self._measurement_identifier_verified:
                if self.measurement_identifier is None:
                    # No measurement identifier was presented by the
                    # referencing dataset. Don't perform any checks.
                    self._measurement_identifier_verified = True
                else:
                    if self.mapping == "same":
                        # When we have identical mapping, then the measurement
                        # identifier has to match exactly.
                        verifier = str.__eq__
                    else:
                        # When we have non-identical mapping (e.g. exported
                        # data), then the measurement identifier has to
                        # partially match.
                        verifier = str.startswith
                    self._measurement_identifier_verified = verifier(
                        self.measurement_identifier,
                        self.get_measurement_identifier()
                    )
            check_rid = self._measurement_identifier_verified
        else:
            check_rid = True

        return check_rid and check_avail


class BasinProxy:
    def __init__(self, ds, basinmap):
        """Proxy for accessing data in basin datasets

        The idea of a basin proxy is to give access to the data of an
        :class:`.RTDCBase` that is mapped, i.e. the indices defined for
        the basin do not coincide with the indices in the downstream
        dataset.

        This class achieves two things:
        1. Subset indexing: For every event in the downstream dataset, there
           is *only* one corresponding event in the basin dataset. This
           could also be achieved via hierarchy children
           (:class:`RTDCHierarchy`).
        2. Blown indexing: Two different events in the downstream dataset
           can refer to one event in the basin dataset. I.e. the basin
           dataset contains fewer events than the downstream dataset,
           because e.g. it is a raw image recording series that has been
           processed and multiple events were found in one frame.

        Parameters
        ----------
        ds: RTDCBase
            the basin dataset
        basinmap: np.ndarray
            1D integer indexing array that maps the events of the basin
            dataset to the downstream dataset
        """
        self.ds = ds
        self.basinmap = basinmap
        self._features = {}

    def __contains__(self, item):
        return item in self.ds

    def __getattr__(self, item):
        if item in [
            "basins",
            "close",
            "features",
            "features_ancillary",
            "features_basin",
            "features_innate",
            "features_loaded",
            "features_local",
            "features_scalar",
            "get_measurement_identifier",
            "ignore_basins",
        ]:
            return getattr(self.ds, item)
        else:
            raise AttributeError(
                f"BasinProxy does not implement {item}")

    def __getitem__(self, feat):
        if feat not in self._features:
            feat_obj = BasinProxyFeature(feat_obj=self.ds[feat],
                                         basinmap=self.basinmap)
            self._features[feat] = feat_obj
        return self._features[feat]

    def __len__(self):
        return len(self.basinmap)


class BasinProxyFeature(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, feat_obj, basinmap):
        """Wrap around a feature object, mapping it upon data access"""
        self.feat_obj = feat_obj
        self.basinmap = basinmap
        self._cache = None
        self.is_scalar = bool(len(self.feat_obj.shape) == 1)

    def __array__(self, dtype=None, *args, **kwargs):
        if self._cache is None and self.is_scalar:
            self._cache = self.feat_obj[:][self.basinmap]
        else:
            # This is dangerous territory in terms of memory usage
            out_arr = np.empty((len(self.basinmap),) + self.feat_obj.shape[1:],
                               dtype=dtype or self.feat_obj.dtype,
                               *args, **kwargs)
            for ii, idx in enumerate(self.basinmap):
                out_arr[ii] = self.feat_obj[idx]
            return out_arr
        return self._cache

    def __getattr__(self, item):
        if item in [
            "dtype",
            "shape",
            "size",
        ]:
            return getattr(self.feat_obj, item)
        else:
            raise AttributeError(
                f"BasinProxyFeature does not implement {item}")

    def __getitem__(self, index):
        if self._cache is None and isinstance(index, numbers.Integral):
            # single index, cheap operation
            return self.feat_obj[self.basinmap[index]]
        elif not self.is_scalar:
            # image, mask, etc
            if isinstance(index, slice) and index == slice(None):
                indices = self.basinmap
            else:
                indices = self.basinmap[index]
            out_arr = np.empty((len(indices),) + self.feat_obj.shape[1:],
                               dtype=self.feat_obj.dtype)
            for ii, idx in enumerate(indices):
                out_arr[ii] = self.feat_obj[idx]
            return out_arr
        else:
            # sets the cache if not already set
            return self.__array__()[index]

    def __len__(self):
        return len(self.basinmap)


def basin_priority_sorted_key(bdict: Dict):
    """Yield a sorting value for a given basin that can be used with `sorted`

    Basins are normally stored in random order in a dataset. This method
    brings them into correct order, prioritizing:

    - type: "file" over "remote"
    - format: "HTTP" over "S3" over "dcor"
    - mapping: "same" over anything else
    """
    srt_type = {
        "internal": "a",
        "file": "b",
        "remote": "c",
    }.get(bdict.get("type"), "z")

    srt_format = {
        "h5dataset": "a",
        "hdf5": "b",
        "http": "c",
        "s3": "d",
        "dcor": "e",
    }.get(bdict.get("format"), "z")

    mapping = bdict.get("mapping", "same")  # old dicts don't have "mapping"
    srt_map = "a" if mapping == "same" else mapping

    return srt_type + srt_format + srt_map


class InternalH5DatasetBasin(Basin):
    basin_format = "h5dataset"
    basin_type = "internal"

    def __init__(self, *args, **kwargs):
        super(InternalH5DatasetBasin, self).__init__(*args, **kwargs)
        if self.mapping == "same":
            raise ValueError(
                "'internal' basins must be instantiated with `mapping`. "
                "If you are not doing that, then you probably don't need "
                "them.")
        if self._features is None:
            raise ValueError("You must specify features when defining "
                             "internal basins.")

    def _load_dataset(self, location, **kwargs):
        from .fmt_dict import RTDC_Dict
        # get the h5file object
        h5root = self._basinmap_referrer().h5file
        ds_dict = {}
        for feat in self.features:
            ds_dict[feat] = h5root[self.location][feat]
        return RTDC_Dict(ds_dict)

    def is_available(self):
        h5root = self._basinmap_referrer().h5file
        return self.location in h5root

    def verify_basin(self, *args, **kwargs):
        """It's not necessary to verify internal basins"""
        return True


def get_basin_classes():
    bc = {}
    for b_cls in Basin.__subclasses__():
        if hasattr(b_cls, "basin_format"):
            bc[b_cls.basin_format] = b_cls
    return bc
