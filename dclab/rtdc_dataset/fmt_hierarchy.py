#!/usr/bin/python
# -*- coding: utf-8 -*-
"""RT-DC hierarchy format"""
from __future__ import division, print_function, unicode_literals

import numpy as np

from .. import definitions as dfn

from .config import Configuration
from .core import RTDCBase
from .filter import Filter
from .util import hashobj


class ChildContour(object):
    def __init__(self, child):
        self.child = child

    def __getitem__(self, idx):
        pidx = map_indices_child2parent(child=self.child,
                                        child_indices=[idx])[0]
        hp = self.child.hparent
        return hp["contour"][pidx]


class ChildImage(object):
    def __init__(self, child):
        self.child = child

    def __getitem__(self, idx):
        pidx = map_indices_child2parent(child=self.child,
                                        child_indices=[idx])[0]
        hp = self.child.hparent
        return hp["image"][pidx]


class ChildTrace(object):
    def __init__(self, child, flname):
        self.child = child
        self.flname = flname

    def __getitem__(self, idx):
        pidx = map_indices_child2parent(child=self.child,
                                        child_indices=[idx])[0]
        hp = self.child.hparent
        return hp["trace"][self.flname][pidx]


class HierarchyFilter(Filter):
    def __init__(self, *args, **kwargs):
        super(HierarchyFilter, self).__init__(*args, **kwargs)
        self.manual_indices = []
        self.parent_hash = hashobj(self.rtdc_ds.hparent.filter.all)

    @property
    def parent_changed(self):
        return hashobj(self.rtdc_ds.hparent.filter.all) != self.parent_hash

    def apply_manual_indices(self, manual_indices):
        """write to self.manual"""
        if self.parent_changed:
            # ignore
            pass
        else:
            self.manual_indices = list(manual_indices)
            cidx = map_indices_root2child(child=self.rtdc_ds,
                                          root_indices=manual_indices)
            if len(cidx):
                self.manual[cidx] = False

    def retrieve_manual_indices(self):
        """read from self.manual"""
        if self.parent_changed:
            # ignore
            pass
        else:
            pidx = map_indices_child2root(
                child=self.rtdc_ds,
                child_indices=np.where(~self.manual)[0])
            # retrieve all indices that are currently not visible
            # previous indices
            pold = self.manual_indices
            pvis = map_indices_root2child(
                child=self.rtdc_ds, root_indices=pold).tolist()
            phid = list(set(pold) - set(pvis))
            all_idx = list(set(phid + pidx.tolist()))

            self.manual_indices = sorted(all_idx)
        return self.manual_indices


class RTDC_Hierarchy(RTDCBase):
    def __init__(self, hparent, *args, **kwargs):
        """A hierarchy child of a subclass of RTDCBase

        A few words on hierarchies:
        The idea is that a subclass of RTDCBase can use the filtered data of another
        subclass of RTDCBase and interpret these data as unfiltered events. This comes
        in handy e.g. when the percentage of different subpopulations need to
        be distinguished without the noise in the original data.

        Children in hierarchies always update their data according to the
        filtered event data from their parent when `apply_filter` is called.
        This makes it easier to save and load hierarchy children with e.g.
        ShapeOut and it makes the handling of hierarchies more intuitive
        (when the parent changes, the child changes as well).

        Parameters
        ----------
        hparent : instance of RTDCBase
            The hierarchy parent.
        *args, **kwargs:
            (Keyword) arguments for `RTDCBase`

        Attributes
        ----------
        hparent : instance of RTDCBase
            Only hierarchy children have this attribute
        """
        super(RTDC_Hierarchy, self).__init__(*args, **kwargs)

        self.path = hparent.path
        self.title = hparent.title + "_child"

        #: hierarchy parent
        self.hparent = hparent

        # Copy configuration
        cfg = hparent.config.copy()

        # Remove previously applied filters
        pops = []
        for key in cfg["filtering"]:
            if (key.endswith("min") or
                key.endswith("max") or
                    key == "polygon filters"):
                pops.append(key)

        [cfg["filtering"].pop(key) for key in pops]
        # Add parent information in dictionary
        cfg["filtering"]["hierarchy parent"] = hparent.identifier

        self.config = Configuration(cfg=cfg)

        # Apply the filter
        # This will also populate all event attributes
        self.apply_filter()

    def __contains__(self, key):
        return self.hparent.__contains__(key)

    def __getitem__(self, key):
        # contour, image, and traces are added automatically
        # to `self._events` in `self.apply_filter`.
        if key not in self._events:
            item = self.hparent[key]
            if key in dfn.feature_names:
                self._events[key] = item[self.hparent._filter]
        return self._events[key]

    def __len__(self):
        return np.sum(self.hparent._filter)

    def _init_filters(self):
        if hasattr(self, "filter"):
            # get manual filters
            manual_pidx = self.filter.retrieve_manual_indices()
        else:
            manual_pidx = []
        # clear filters
        super(RTDC_Hierarchy, self)._init_filters()
        # override standard filter
        self.filter = HierarchyFilter(self)

        if len(manual_pidx):
            # set manually excluded events
            self.filter.apply_manual_indices(manual_pidx)

    def apply_filter(self, *args, **kwargs):
        """Overridden `apply_filter` to perform tasks for hierarchy child"""
        if hasattr(self, "filter"):
            # make sure self.filter knows about root manual indices
            self.filter.retrieve_manual_indices()

        # Copy event data from hierarchy parent
        self.hparent.apply_filter()
        # update event index
        event_count = np.sum(self.hparent._filter)
        self._events = {}
        self._events["index"] = np.arange(1, event_count + 1)
        # set non-scalar column data
        if "contour" in self.hparent:
            self._events["contour"] = ChildContour(self)
        if "image" in self.hparent:
            self._events["image"] = ChildImage(self)
        if "trace" in self.hparent:
            trdict = {}
            for flname in dfn.FLUOR_TRACES:
                if flname in self.hparent["trace"]:
                    trdict[flname] = ChildTrace(self, flname)
            self._events["trace"] = trdict
        # update config
        self.config["experiment"]["event count"] = event_count

        self._init_filters()
        super(RTDC_Hierarchy, self).apply_filter(*args, **kwargs)

    @property
    def hash(self):
        """Hashes of a hierarchy child changes if the parent changes"""
        # Do not apply filters here (speed)
        hph = self.hparent.hash
        hpfilt = hashobj(self.hparent._filter)
        dhash = hashobj(hph + hpfilt)
        return dhash


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
