from __future__ import annotations

import numbers

import numpy as np

from ...util import copy_if_needed


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
        self.basins_get_dicts = ds.basins_get_dicts
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
            if feat == "contour":
                feat_obj = BasinProxyContour(feat_obj=self.ds[feat],
                                             basinmap=self.basinmap)
            else:
                feat_obj = BasinProxyFeature(feat_obj=self.ds[feat],
                                             basinmap=self.basinmap)
            self._features[feat] = feat_obj
        return self._features[feat]

    def __len__(self):
        return len(self.basinmap)


class BasinProxyContour:
    def __init__(self, feat_obj, basinmap):
        """Wrap around a contour, mapping it upon data access, no caching"""
        self.feat_obj = feat_obj
        self.basinmap = basinmap
        self.is_scalar = False
        self.shape = (len(self.basinmap), np.nan, 2)
        self.identifier = feat_obj.identifier

    def __getattr__(self, item):
        if item in [
            "dtype",
        ]:
            return getattr(self.feat_obj, item)
        else:
            raise AttributeError(
                f"BasinProxyContour does not implement {item}")

    def __getitem__(self, index):
        if isinstance(index, numbers.Integral):
            # single index, cheap operation
            return self.feat_obj[self.basinmap[index]]
        else:
            raise NotImplementedError(
                "Cannot index contours without anything else than integers.")

    def __len__(self):
        return self.shape[0]


class BasinProxyFeature(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, feat_obj, basinmap):
        """Wrap around a feature object, mapping it upon data access"""
        self.feat_obj = feat_obj
        self.basinmap = basinmap
        self._cache = None
        self._shape = None
        self._size = None
        self.is_scalar = bool(len(self.feat_obj.shape) == 1)

    @property
    def shape(self):
        if self._shape is None:
            if self.is_scalar:
                self._shape = self.basinmap.shape
            else:
                self._shape = (self.basinmap.size,) + self.feat_obj.shape[1:]
        return self._shape

    @property
    def size(self):
        if self._size is None:
            self._size = np.prod(self.shape)
        return self._size

    def __array__(self, dtype=None, copy=copy_if_needed, *args, **kwargs):
        if self._cache is None and self.is_scalar:
            self._cache = self.feat_obj[:][self.basinmap]
            return np.array(self._cache, copy=copy)
        else:
            # This is dangerous territory in terms of memory usage
            out_arr = np.empty((len(self.basinmap),) + self.feat_obj.shape[1:],
                               *args,
                               dtype=dtype or self.feat_obj.dtype,
                               **kwargs)
            for ii, idx in enumerate(self.basinmap):
                out_arr[ii] = self.feat_obj[idx]
            return out_arr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Convert all instances of `BasinProxyFeature` to arrays.
        inputs = [ip.__array__() if isinstance(ip, BasinProxyFeature) else ip
                  for ip in inputs]
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __getattr__(self, item):
        if item in [
            "dtype",
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

    def max(self, *args, **kwargs):
        if self.is_scalar:
            if np.all(self.basinmap):
                # If the original basin dataset has the ufunc specified
                # as an HDF5 attribute or similar, then this is faster.
                return self.feat_obj.max()
            else:
                # Compute the maximum at the cost of potentially having
                # to download the data.
                return np.max(self.feat_obj[self.basinmap])
        else:
            raise NotImplementedError(
                f"ufunc 'max' only available for scalar features in "
                f"'{self.__class__.__name__}'")

    def mean(self, *args, **kwargs):
        if self.is_scalar:
            if np.all(self.basinmap):
                # If the original basin dataset has the ufunc specified
                # as an HDF5 attribute or similar, then this is faster.
                return self.feat_obj.mean()
            else:
                # Compute the mean at the cost of potentially having
                # to download the data.
                return np.mean(self.feat_obj[self.basinmap])
        else:
            raise NotImplementedError(
                f"ufunc 'mean' only available for scalar features in "
                f"'{self.__class__.__name__}'")

    def min(self, *args, **kwargs):
        if self.is_scalar:
            if np.all(self.basinmap):
                # If the original basin dataset has the ufunc specified
                # as an HDF5 attribute or similar, then this is faster.
                return self.feat_obj.min()
            else:
                # Compute the minimum at the cost of potentially having
                # to download the data.
                return np.min(self.feat_obj[self.basinmap])
        else:
            raise NotImplementedError(
                f"ufunc 'min' only available for scalar features in "
                f"'{self.__class__.__name__}'")
