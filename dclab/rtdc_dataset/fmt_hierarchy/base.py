"""RT-DC hierarchy format"""
import numpy as np

from ... import definitions as dfn

from ...util import hashobj

from ..config import Configuration
from ..core import RTDCBase

from .events import (
    ChildContour, ChildNDArray, ChildScalar, ChildTrace, ChildTraceItem
)
from .hfilter import HierarchyFilter


class RTDC_Hierarchy(RTDCBase):
    def __init__(self, hparent, apply_filter=True, *args, **kwargs):
        """Hierarchy dataset (filtered from RTDCBase)

        A few words on hierarchies:
        The idea is that a subclass of RTDCBase can use the filtered
        data of another subclass of RTDCBase and interpret these data
        as unfiltered events. This comes in handy e.g. when the
        percentage of different subpopulations need to be distinguished
        without the noise in the original data.

        Children in hierarchies always update their data according to
        the filtered event data from their parent when `apply_filter`
        is called. This makes it easier to save and load hierarchy
        children with e.g. Shape-Out and it makes the handling of
        hierarchies more intuitive (when the parent changes,
        the child changes as well).

        Parameters
        ----------
        hparent: instance of RTDCBase
            The hierarchy parent
        apply_filter: bool
            Whether to apply the filter during instantiation;
            If set to `False`, `apply_filter` must be called
            manually.
        *args:
            Arguments for `RTDCBase`
        **kwargs:
            Keyword arguments for `RTDCBase`

        Attributes
        ----------
        hparent: RTDCBase
            Hierarchy parent of this instance
        """
        super(RTDC_Hierarchy, self).__init__(*args, **kwargs)

        self.path = hparent.path
        self.title = hparent.title + "_child"
        self.logs = hparent.logs
        self.tables = hparent.tables

        self._events = {}

        #: hierarchy parent
        self.hparent = hparent

        self.config = self._create_config()  # init config
        self._update_config()  # sets e.g. event count

        if apply_filter:
            # Apply the filter
            # This will also populate all event attributes
            self.apply_filter()

        self._length = None

    def __contains__(self, key):
        return self.hparent.__contains__(key)

    def __getitem__(self, feat):
        """Return the feature data and cache them in self._events"""
        if feat in self._events:
            data = self._events[feat]
        elif feat in self.hparent:
            if len(self.hparent[feat].shape) > 1:
                # non-scalar feature
                data = ChildNDArray(self, feat)
            else:
                # scalar feature
                data = ChildScalar(self, feat)
            # Cache everything, even the Young's modulus. The user is
            # responsible for calling `rejuvenate` to reset everything.
            self._events[feat] = data
        else:
            raise KeyError(
                f"The dataset {self} does not contain the feature '{feat}'! "
                + "If you are attempting to access an ancillary feature "
                + "(e.g. emodulus), please make sure that the feature "
                + f"data are computed for {self.get_root_parent()} (the "
                + "root parent of this hierarchy child).")
        return data

    def __len__(self):
        if self._length is None:
            self._length = np.sum(self.hparent.filter.all)
        return self._length

    def _assert_filter(self):
        """Make sure filters exists

        Override from base class that uses :class:`.HierarchyFilter`
        instead of :class:`.Filter`.
        """
        if self._ds_filter is None:
            self._ds_filter = HierarchyFilter(self)

    def _check_parent_filter(self):
        """Reset filter if parent changed

        This will create a new HierarchyFilter for self if the
        parent RTDCBase changed. We do it like this, because it
        would be complicated to track all the changes in
        HierarchyFilter.
        """
        if self.filter.parent_changed:
            manual_pidx = self.filter.retrieve_manual_indices(self)
            self._ds_filter = None  # forces recreation of HierarchyFilter
            self._assert_filter()
            self.filter.apply_manual_indices(self, manual_pidx)

    def _create_config(self):
        """Return a stripped configuration from the parent"""
        # create a new configuration
        cfg = self.hparent.config.copy()
        # Remove previously applied filters
        pops = []
        for key in cfg["filtering"]:
            if (key.endswith(" min") or
                key.endswith(" max") or
                    key == "polygon filters"):
                pops.append(key)
        [cfg["filtering"].pop(key) for key in pops]
        # Add parent information in dictionary
        cfg["filtering"]["hierarchy parent"] = self.hparent.identifier
        return Configuration(cfg=cfg)

    def _update_config(self):
        """Update varying config values from self.hparent"""
        # event count
        self.config["experiment"]["event count"] = np.sum(
            self.hparent.filter.all)
        # calculation
        if "calculation" in self.hparent.config:
            self.config["calculation"].clear()
            self.config["calculation"].update(
                self.hparent.config["calculation"])

    @property
    def features(self):
        return self.hparent.features

    @property
    def features_innate(self):
        return self.hparent.features_innate

    @property
    def features_loaded(self):
        return self.hparent.features_loaded

    @property
    def features_scalar(self):
        return self.hparent.features_scalar

    @property
    def hash(self):
        """Hashes of a hierarchy child changes if the parent changes"""
        # Do not apply filters here (speed)
        hph = self.hparent.hash
        hpfilt = hashobj(self.hparent.filter.all)
        dhash = hashobj(hph + hpfilt)
        return dhash

    def apply_filter(self, *args, **kwargs):
        """Overridden `apply_filter` to perform tasks for hierarchy child"""
        if self._ds_filter is not None:
            # make sure self.filter knows about root manual indices
            # (stored in self.filter._man_root_ids)
            self.filter.retrieve_manual_indices(self)

        # Copy event data from hierarchy parent
        self.hparent.apply_filter(*args, **kwargs)

        # Clear anything that has been cached until now
        self._length = None

        # update event index
        event_count = len(self)
        self._events.clear()
        self._events["index"] = np.arange(1, event_count + 1)
        # set non-scalar column data
        for feat in ["image", "image_bg", "mask"]:
            if feat in self.hparent:
                self._events[feat] = ChildNDArray(self, feat)
        if "contour" in self.hparent:
            self._events["contour"] = ChildContour(self)
        if "trace" in self.hparent:
            trdict = ChildTrace()
            for flname in dfn.FLUOR_TRACES:
                if flname in self.hparent["trace"]:
                    trdict[flname] = ChildTraceItem(self, flname)
            self._events["trace"] = trdict
        # Update configuration
        self._update_config()

        # create a new filter if the parent changed
        self._check_parent_filter()
        super(RTDC_Hierarchy, self).apply_filter(*args, **kwargs)

    def get_root_parent(self):
        """Return the root parent of this dataset"""
        if isinstance(self.hparent, RTDC_Hierarchy):
            return self.hparent.get_root_parent()
        else:
            return self.hparent

    def rejuvenate(self):
        """Redraw the hierarchy tree, updating config and features

        You should call this function whenever you change something
        in the hierarchy parent(s), be it filters or metadata for computing
        ancillary features.

        .. versionadded: 0.47.0
            This is just an alias of `apply_filter`, but with a more
            accurate name (not only the filters are applied, but alot
            of other things might change).
        """
        self.apply_filter()
