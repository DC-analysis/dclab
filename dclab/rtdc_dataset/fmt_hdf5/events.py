"""RT-DC hdf5 format"""
from __future__ import annotations

import warnings

import numbers
import numpy as np

from ... import definitions as dfn


from . import feat_defect


class H5ContourEvent:
    def __init__(self, h5group, length=None):
        self._length = length
        self.h5group = h5group
        # for hashing in util.obj2bytes
        self.identifier = (h5group.file.filename, h5group["0"].name)

    def __getitem__(self, key):
        if not isinstance(key, numbers.Integral):
            # slicing!
            indices = np.arange(len(self))[key]
            output = []
            # populate the output list
            for evid in indices:
                output.append(self.h5group[str(evid)][:])
            return output
        elif key < 0:
            return self.__getitem__(key + len(self))
        else:
            return self.h5group[str(key)][:]

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self):
        if self._length is None:
            # computing the length of an H5Group is slow
            self._length = len(self.h5group)
        return self._length

    @property
    def dtype(self):
        return self.h5group["0"].dtype

    @property
    def shape(self):
        return len(self), np.nan, 2


class H5Events:
    def __init__(self, h5):
        self.h5file = h5
        # According to https://github.com/h5py/h5py/issues/1960, we always
        # have to keep a reference to the HDF5 dataset, otherwise it will
        # be garbage-collected immediately. In addition to caching the HDF5
        # datasets, we cache the wrapping classes in the `self._cached_events`
        # dictionary.
        self._cached_events = {}
        self._defective_features = {}
        self._features_list = None

    @property
    def _features(self):
        if self._features_list is None:
            self._features_list = sorted(self.h5file["events"].keys())
            # make sure that "trace" is not empty
            if ("trace" in self._features
                    and len(self.h5file["events"]["trace"]) == 0):
                self._features_list.remove("trace")
        return self._features_list

    def __contains__(self, key):
        return key in self.keys()

    def __getitem__(self, key):
        if key not in self._cached_events:
            # user-level checking is done in core.py
            assert dfn.feature_exists(key), f"Feature '{key}' does not exist!"
            data = self.h5file["events"][key]
            if key == "contour":
                length = self.h5file.attrs.get("experiment:event count")
                fdata = H5ContourEvent(data, length=length)
            elif key == "mask":
                fdata = H5MaskEvent(data)
            elif key == "trace":
                fdata = H5TraceEvent(data)
            elif data.ndim == 1:
                fdata = H5ScalarEvent(data)
            else:
                # for features like "image", "image_bg" and other non-scalar
                # ancillary features
                fdata = data
            self._cached_events[key] = fdata
        return self._cached_events[key]

    def __iter__(self):
        # dict-like behavior
        for key in self.keys():
            yield key

    def _is_defective_feature(self, feat):
        """Whether the stored feature is defective"""
        if feat not in self._defective_features:
            defective = False
            if (feat in feat_defect.DEFECTIVE_FEATURES
                    and feat in self._features):
                # feature exists in the HDF5 file
                # workaround machinery for sorting out defective features
                defective = feat_defect.DEFECTIVE_FEATURES[feat](self.h5file)
            self._defective_features[feat] = defective
        return self._defective_features[feat]

    def keys(self):
        """Returns list of valid features

        Checks for
        - defective features: whether the data in the HDF5 file is invalid
        - existing feature names: dynamic, depending on e.g. plugin features
        """
        features = []
        for key in self._features:
            # check for defective features
            if dfn.feature_exists(key) and not self._is_defective_feature(key):
                features.append(key)
        return features


class H5MaskEvent:
    """Cast uint8 masks to boolean"""

    def __init__(self, h5dataset):
        self.h5dataset = h5dataset
        # identifier required because "mask" is used for computation
        # of ancillary feature "contour".
        self.identifier = (self.h5dataset.file.filename, self.h5dataset.name)
        self.dtype = np.dtype(bool)

    def __array__(self, dtype=np.bool_):
        if dtype is not np.uint8:
            warnings.warn("Please avoid calling the `__array__` method of the "
                          "`H5MaskEvent`. It may consume a lot of memory.",
                          UserWarning)
        # One of the reasons why we implement __array__ is such that
        # the data exporter knows this object is sliceable
        # (see yield_filtered_array_stacks).
        return self.h5dataset.__array__(dtype=dtype)

    def __getitem__(self, idx):
        return np.asarray(self.h5dataset[idx], dtype=bool)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self):
        return len(self.h5dataset)

    @property
    def attrs(self):
        return self.h5dataset.attrs

    @property
    def shape(self):
        return self.h5dataset.shape


class H5ScalarEvent(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, h5ds):
        """Lazy access to a scalar feature with cache"""
        self.h5ds = h5ds
        # for hashing in util.obj2bytes
        self.identifier = (self.h5ds.file.filename, self.h5ds.name)
        self._array = None
        self.ndim = 1  # matplotlib might expect this from an array
        # attrs
        self._ufunc_attrs = dict(self.h5ds.attrs)

    def __array__(self, dtype=None):
        if self._array is None:
            self._array = np.asarray(self.h5ds, dtype=dtype)
        return self._array

    def __getitem__(self, idx):
        return self.__array__()[idx]

    def __len__(self):
        return len(self.h5ds)

    def _fetch_ufunc_attr(self, uname, ufunc):
        """A wrapper for calling functions on the scalar feature data

        The ideas are:

        1. If there is a ufunc (max/mean/min) value stored in the dataset
           attributes, then use this one.
        2. If the ufunc is computed, it is cached permanently in
           self._ufunc_attrs
        """
        val = self._ufunc_attrs.get(uname, None)
        if val is None:
            val = ufunc(self.__array__())
            self._ufunc_attrs[uname] = val
        return val

    def max(self, *args, **kwargs):
        return self._fetch_ufunc_attr("max", np.nanmax)

    def mean(self, *args, **kwargs):
        return self._fetch_ufunc_attr("mean", np.nanmean)

    def min(self, *args, **kwargs):
        return self._fetch_ufunc_attr("min", np.nanmin)

    @property
    def shape(self):
        return self.h5ds.shape


class H5TraceEvent:
    def __init__(self, h5group):
        self.h5group = h5group
        self._num_traces = None
        self._shape = None

    def __getitem__(self, idx):
        return self.h5group[idx]

    def __contains__(self, item):
        return item in self.h5group

    def __len__(self):
        if self._num_traces is None:
            self._num_traces = len(self.h5group)
        return self._num_traces

    def __iter__(self):
        for key in sorted(self.h5group.keys()):
            yield key

    def keys(self):
        return self.h5group.keys()

    @property
    def shape(self):
        if self._shape is None:
            atrace = list(self.h5group.keys())[0]
            self._shape = tuple([len(self)] + list(self.h5group[atrace].shape))
        return self._shape
