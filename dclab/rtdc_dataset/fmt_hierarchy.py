"""RT-DC hierarchy format"""

import numpy as np

from .. import definitions as dfn

from ..util import hashobj

from .config import Configuration
from .core import RTDCBase
from .filter import Filter


class HierarchyFilterError(BaseException):
    """Used for unexpected filtering operations"""
    pass


class ChildBase(object):
    def __init__(self, child):
        self.child = child

    def __len__(self):
        return len(self.child)


class ChildContour(ChildBase):
    def __getitem__(self, idx):
        pidx = map_indices_child2parent(child=self.child,
                                        child_indices=idx)
        hp = self.child.hparent
        return hp["contour"][pidx]


class ChildImage(ChildBase):
    def __getitem__(self, idx):
        pidx = map_indices_child2parent(child=self.child,
                                        child_indices=idx)
        hp = self.child.hparent
        return hp["image"][pidx]


class ChildImageBG(ChildBase):
    def __getitem__(self, idx):
        pidx = map_indices_child2parent(child=self.child,
                                        child_indices=idx)
        hp = self.child.hparent
        return hp["image_bg"][pidx]


class ChildMask(ChildBase):
    def __getitem__(self, idx):
        pidx = map_indices_child2parent(child=self.child,
                                        child_indices=idx)
        hp = self.child.hparent
        return hp["mask"][pidx]


class ChildTrace(ChildBase):
    def __init__(self, child, flname):
        super(ChildTrace, self).__init__(child)
        self.flname = flname

    def __getitem__(self, idx):
        pidx = map_indices_child2parent(child=self.child,
                                        child_indices=idx)
        hp = self.child.hparent
        return hp["trace"][self.flname][pidx]


class HierarchyFilter(Filter):
    def __init__(self, rtdc_ds, *args, **kwargs):
        """A filtering class for RTDC_Hierarchy

        This subclass handles manual filters for hierarchy children.
        The general problem with hierarchy children is that their data
        changes when the hierarchy parent changes. As hierarchy
        children may also have hierarchy children, dealing with
        manual filters (`Filter.manual`) is not trivial. Here,
        the manual filters are translated into event indices of the
        root parent (the highest member of the hierarchy, which is
        `RTDC_Hierarchy.hparent` if there is only one child).
        This enables to keep track of the manually excluded events
        even if

        - the parent changes its filters,
        - the parent is a hierarchy child as well, or
        - the excluded event is filtered out in the parent.
        """
        super(HierarchyFilter, self).__init__(rtdc_ds, *args, **kwargs)
        self.update_parent(rtdc_ds.hparent)
        self._man_root_ids = []

    @property
    def parent_changed(self):
        return hashobj(self._parent_rtdc_ds.filter.all) != self._parent_hash

    def apply_manual_indices(self, rtdc_ds, manual_indices):
        """Write to `self.manual`

        Write `manual_indices` to the boolean array `self.manual`
        and also store the indices as `self._man_root_ids`.

        Notes
        -----
        If `self.parent_changed` is `True`, i.e. the parent applied
        a filter and the child did not yet hear about this, then
        `HierarchyFilterError` is raised. This is important, because
        the size of the current filter would not match the size of
        the filtered events of the parent and thus index-mapping
        would not work.
        """
        if self.parent_changed:
            msg = "Cannot apply filter, because parent changed: " \
                  + "dataset {}. ".format(rtdc_ds) \
                  + "Run `RTDC_Hierarchy.apply_filter()` first!"
            raise HierarchyFilterError(msg)
        else:
            self._man_root_ids = list(manual_indices)
            cidx = map_indices_root2child(child=rtdc_ds,
                                          root_indices=manual_indices)
            if len(cidx):
                self.manual[cidx] = False

    def reset(self):
        super(HierarchyFilter, self).reset()
        self._man_root_ids = []

    def retrieve_manual_indices(self, rtdc_ds):
        """Read from self.manual

        Read from the boolean array `self.manual`, index all
        occurences of `False` and find the corresponding indices
        in the root hierarchy parent, return those and store them
        in `self._man_root_ids` as well.

        Notes
        -----
        This method also retrieves hidden indices, i.e. events
        that are not part of the current hierarchy child but
        which have been manually excluded before and are now
        hidden because a hierarchy parent filtered it out.

        If `self.parent_changed` is `True`, i.e. the parent applied
        a filter and the child did not yet hear about this, then
        nothing is computed and `self._man_root_ids` as-is.  This
        is important, because the size of the current filter would
        not match the size of the filtered events of the parent and
        thus index-mapping would not work.
        """
        if self.parent_changed:
            # ignore
            pass
        else:
            # indices from boolean array
            pbool = map_indices_child2root(
                child=rtdc_ds,
                child_indices=np.where(~self.manual)[0]).tolist()
            # retrieve all indices that are currently not visible
            # previous indices
            pold = self._man_root_ids
            # all indices previously selected either via
            # - self.manual or
            # - self.apply_manual_indices
            pall = sorted(list(set(pbool + pold)))
            # visible indices (only available child indices are returned)
            pvis_c = map_indices_root2child(child=rtdc_ds,
                                            root_indices=pall).tolist()
            # map visible child indices back to root indices
            pvis_p = map_indices_child2root(child=rtdc_ds,
                                            child_indices=pvis_c).tolist()
            # hidden indices
            phid = list(set(pall) - set(pvis_p))
            # Why not set `all_idx` to `pall`:
            # - pbool is considered to be correct
            # - pold contains hidden indices, but also might contain
            #   excess indices from before, i.e. if self.apply_manual_indices
            #   is called, self.manual is also updated. If however,
            #   self.manual is updated, self._man_root_ids are not updated.
            #   Thus, we trust pbool (self.manual) and only use pold
            #   (self._man_root_ids) to determine hidden indices.
            all_idx = list(set(pbool + phid))
            self._man_root_ids = sorted(all_idx)
        return self._man_root_ids

    def update_parent(self, parent_rtdc_ds):
        # hold reference to rtdc_ds parent
        # (not to its filter, because that is reinstantiated)
        self._parent_rtdc_ds = parent_rtdc_ds
        self._parent_hash = hashobj(self._parent_rtdc_ds.filter.all)


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

        #: hierarchy parent
        self.hparent = hparent

        self.filter = HierarchyFilter(self)

        self.config = self._create_config()  # init config
        self._update_config()  # sets e.g. event count

        if apply_filter:
            # Apply the filter
            # This will also populate all event attributes
            self.apply_filter()

    def __contains__(self, key):
        return self.hparent.__contains__(key)

    def __getitem__(self, key):
        # contour, image, and traces are added automatically
        # to `self._events` in `self.apply_filter`.
        if key in self._events:
            return self._events[key]
        else:
            item = self.hparent[key]
            return item[self.hparent.filter.all]

    def __len__(self):
        return np.sum(self.hparent.filter.all)

    def _check_parent_filter(self):
        """Reset filter if parent changed

        This will create a new HierarchyFilter for self if the
        parent RTDCBase changed. We do it like this, because it
        would be complicated to track all the changes in
        HierarchyFilter.
        """
        if self.filter.parent_changed:
            manual_pidx = self.filter.retrieve_manual_indices(self)
            self.filter = HierarchyFilter(self)
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
        if self.filter is not None:
            # make sure self.filter knows about root manual indices
            # (stored in self.filter._man_root_ids)
            self.filter.retrieve_manual_indices(self)

        # Copy event data from hierarchy parent
        self.hparent.apply_filter(*args, **kwargs)
        # update event index
        event_count = np.sum(self.hparent.filter.all)
        self._events = {}
        self._events["index"] = np.arange(1, event_count + 1)
        # set non-scalar column data
        if "contour" in self.hparent:
            self._events["contour"] = ChildContour(self)
        if "image" in self.hparent:
            self._events["image"] = ChildImage(self)
        if "image_bg" in self.hparent:
            self._events["image_bg"] = ChildImageBG(self)
        if "mask" in self.hparent:
            self._events["mask"] = ChildMask(self)
        if "trace" in self.hparent:
            trdict = {}
            for flname in dfn.FLUOR_TRACES:
                if flname in self.hparent["trace"]:
                    trdict[flname] = ChildTrace(self, flname)
            self._events["trace"] = trdict
        # Update configuration
        self._update_config()

        # create a new filter if the parent changed
        self._check_parent_filter()
        super(RTDC_Hierarchy, self).apply_filter(*args, **kwargs)


def map_indices_child2parent(child, child_indices):
    """Map child RTDCBase event indices to parent RTDCBase

    Parameters
    ----------
    child: RTDC_Hierarchy
        hierarchy child with `child_indices`
    child_indices: 1d ndarray
        child indices to map

    Returns
    -------
    parent_indices: 1d ndarray
        hierarchy parent indices
    """
    parent = child.hparent
    # filters
    pf = parent.filter.all
    # indices corresponding to all child events
    idx = np.where(pf)[0]  # True means present in the child
    # indices corresponding to selected child events
    parent_indices = idx[child_indices]
    return parent_indices


def map_indices_child2root(child, child_indices):
    """Map RTDC_Hierarchy event indices to root RTDCBase

    Parameters
    ----------
    child: RTDC_Hierarchy
        RTDCBase hierarchy child
    child_indices: 1d ndarray
        child indices to map

    Returns
    -------
    root_indices: 1d ndarray
        hierarchy root indices
        (not necessarily the indices of `parent`)
    """
    while True:
        indices = map_indices_child2parent(child=child,
                                           child_indices=child_indices)
        if isinstance(child.hparent, RTDC_Hierarchy):
            child = child.hparent
            child_indices = indices
        else:
            break
    return indices


def map_indices_parent2child(child, parent_indices):
    """Map parent RTDCBase event indices to RTDC_Hierarchy

    Parameters
    ----------
    parent: RTDC_Hierarchy
        hierarchy child
    parent_indices: 1d ndarray
        hierarchy parent (`child.hparent`) indices to map

    Returns
    -------
    child_indices: 1d ndarray
        child indices
    """
    parent = child.hparent
    # filters
    pf = parent.filter.all
    # indices in child
    child_indices = []
    count = 0
    for ii in range(len(pf)):
        if pf[ii]:
            # only append indices if they exist in child
            if ii in parent_indices:
                # current child event count is the child index
                child_indices.append(count)
            # increment child event count
            count += 1

    return np.array(child_indices)


def map_indices_root2child(child, root_indices):
    """Map root RTDCBase event indices to child RTDCBase

    Parameters
    ----------
    parent: RTDCBase
        hierarchy parent of `child`.
    root_indices: 1d ndarray
        hierarchy root indices to map
        (not necessarily the indices of `parent`)

    Returns
    -------
    child_indices: 1d ndarray
        child indices
    """
    # construct hierarchy tree containing only RTDC_Hierarchy instances
    hierarchy = [child]
    while True:
        if isinstance(child.hparent, RTDC_Hierarchy):
            # the parent is a hierarchy tree
            hierarchy.append(child.hparent)
            child = child.hparent
        else:
            break

    indices = root_indices
    for hp in hierarchy[::-1]:  # reverse order
        # For each hierarchy parent, map the indices down the
        # hierarchy tree.
        indices = map_indices_parent2child(child=hp,
                                           parent_indices=indices)
    return indices
