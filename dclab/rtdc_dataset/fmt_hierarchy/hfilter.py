import numpy as np

from ...util import hashobj

from ..filter import Filter

from .mapper import map_indices_root2child, map_indices_child2root


class HierarchyFilterError(BaseException):
    """Used for unexpected filtering operations"""
    pass


class HierarchyFilter(Filter):
    def __init__(self, rtdc_ds):
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
        self._man_root_ids = []
        super(HierarchyFilter, self).__init__(rtdc_ds)
        self._parent_rtdc_ds = None
        self._parent_hash = None
        self.update_parent(rtdc_ds.hparent)

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
        self._man_root_ids.clear()

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
        elif np.all(self.manual):
            # Do not do anything and remember the events we manually
            # excluded in case the parent reinserts them.
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
