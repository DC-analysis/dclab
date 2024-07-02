import collections
import warnings

import numpy as np

from .mapper import map_indices_child2parent


class ChildBase(object):
    def __init__(self, child):
        self.child = child

    def __len__(self):
        return len(self.child)


class ChildContour(ChildBase):
    def __init__(self, child):
        super(ChildContour, self).__init__(child)
        self.shape = (len(child), np.nan, 2)
        # Note that since the contour has variable lengths, we cannot
        # implement an __array__ method here.

    def __getitem__(self, idx):
        pidx = map_indices_child2parent(child=self.child,
                                        child_indices=idx)
        hp = self.child.hparent
        return hp["contour"][pidx]

    @property
    def dtype(self):
        return self.child.hparent["contour"].dtype


class ChildNDArray(ChildBase):
    def __init__(self, child, feat):
        super(ChildNDArray, self).__init__(child)
        self.feat = feat

    def __array__(self, *args, **kwargs):
        warnings.warn("Please avoid calling the `__array__` method of the "
                      "`ChildNDArray`. It may consume a lot of memory. "
                      "Consider using a generator instead.",
                      UserWarning)
        return np.asarray(self[:], *args, **kwargs)

    def __getitem__(self, idx):
        pidx = map_indices_child2parent(child=self.child,
                                        child_indices=idx)
        hp = self.child.hparent
        return hp[self.feat][pidx]

    @property
    def dtype(self):
        return self.child.hparent[self.feat].dtype

    @property
    def shape(self):
        hp = self.child.hparent
        return tuple([len(self)] + list(hp[self.feat][0].shape))


class ChildScalar(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, child, feat):
        self.child = child
        self.feat = feat
        self._array = None
        # ufunc metadata attribute cache
        self._ufunc_attrs = {}
        self.ndim = 1  # matplotlib might expect this from an array

    def __array__(self, *args, **kwargs):
        if self._array is None:
            hparent = self.child.hparent
            filt_arr = hparent.filter.all
            self._array = hparent[self.feat][filt_arr]
        return np.asarray(self._array, *args, **kwargs)

    def __getitem__(self, idx):
        return self.__array__()[idx]

    def __len__(self):
        return len(self.child)

    def _fetch_ufunc_attr(self, uname, ufunc):
        """A wrapper for calling functions on the scalar feature data

        If the ufunc is computed, it is cached permanently in
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
        return len(self),


class ChildTrace(collections.UserDict):
    @property
    def shape(self):
        # set proper shape (#117)
        key0 = sorted(self.keys())[0]
        return tuple([len(self)] + list(self[key0].shape))


class ChildTraceItem(ChildBase):
    def __init__(self, child, flname):
        super(ChildTraceItem, self).__init__(child)
        self.flname = flname

    def __array__(self, *args, **kwargs):
        warnings.warn("Please avoid calling the `__array__` method of the "
                      "`ChildTraceItem`. It may consume a lot of memory. "
                      "Consider using a generator instead.",
                      UserWarning)
        return np.asarray(self[:], *args, **kwargs)

    def __getitem__(self, idx):
        pidx = map_indices_child2parent(child=self.child,
                                        child_indices=idx)
        hp = self.child.hparent
        return hp["trace"][self.flname][pidx]

    @property
    def dtype(self):
        hp = self.child.hparent
        return hp["trace"][self.flname].dtype

    @property
    def shape(self):
        hp = self.child.hparent
        return len(self), hp["trace"][self.flname].shape[1]
