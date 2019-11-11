#!/usr/bin/python
# -*- coding: utf-8 -*-
"""RT-DC dataset core classes and methods"""
from __future__ import division, print_function, unicode_literals

import warnings

import numpy as np

from dclab import definitions as dfn

from .. import downsampling
from ..polygon_filter import PolygonFilter


class NanWarning(UserWarning):
    pass


class Filter(object):
    def __init__(self, rtdc_ds):
        """Boolean filter arrays for RT-DC measurements

        Parameters
        ----------
        rtdc_ds: instance of RTDCBase
            The RT-DC dataset the filter applies to
        """
        # initialize important parameters
        self._init_rtdc_ds(rtdc_ds)
        # initialize properties
        self.reset()

    def __getitem__(self, key):
        """Return the filter for a feature in `self.features`"""
        if key in self.features and key in dfn.scalar_feature_names:
            if key not in self._filters:
                # Generate filters on-the-fly
                self._filters[key] = np.ones(self.size, dtype=bool)
        else:
            raise KeyError("Feature not available: '{}'".format(key))
        return self._filters[key]

    def _init_rtdc_ds(self, rtdc_ds):
        #: Available feature names
        self.features = rtdc_ds.features
        self.size = len(rtdc_ds)

    def reset(self):
        """Reset all filters"""
        # Box filters
        self._filters = {}
        #: All filters combined (see :func:`Filter.update`);
        #: Use this property to filter the features of
        #: :class:`dclab.rtdc_dataset.RTDCBase` instances
        self.all = np.ones(self.size, dtype=bool)
        #: Invalid (nan/inf) events
        self.invalid = np.ones(self.size, dtype=bool)
        #: 1D boolean array for manually excluding events; `False` values
        #: are excluded.
        self.manual = np.ones(self.size, dtype=bool)
        #: Polygon filters; Note that this array may only contain `True`
        #: values at positions where all other filters are `True`
        #: (this is by design to save computation time).
        self.polygon = np.ones(self.size, dtype=bool)
        # old filter configuration of `rtdc_ds`
        self._old_config = {}

    def update(self, rtdc_ds, force=[]):
        """Update the filters according to `rtdc_ds.config["filtering"]`

        Parameters
        ----------
        rtdc_ds: dclab.rtdc_dataset.core.RTDCBase
            The measurement to which the filter is applied
        force : list
            A list of feature names that must be refiltered with
            min/max values.

        Notes
        -----
        This function is called when
        :func:`ds.apply_filter <dclab.rtdc_dataset.RTDCBase.apply_filter>`
        is called.
        """
        # re-initialize important parameters
        self._init_rtdc_ds(rtdc_ds)

        # These lists may help us become very fast in the future
        newkeys = []
        oldvals = []
        newvals = []

        cfg_cur = rtdc_ds.config["filtering"]
        cfg_old = self._old_config

        # Determine which data was updated
        for skey in list(cfg_cur.keys()):
            if skey not in cfg_old:
                cfg_old[skey] = None
            if cfg_cur[skey] != cfg_old[skey]:
                newkeys.append(skey)
                oldvals.append(cfg_old[skey])
                newvals.append(cfg_cur[skey])

        # 1. Invalid filters
        self.invalid[:] = True
        if cfg_cur["remove invalid events"]:
            for feat in dfn.scalar_feature_names:
                if feat in rtdc_ds:
                    data = rtdc_ds[feat]
                    invalid = np.isinf(data) | np.isnan(data)
                    self.invalid &= ~invalid

        # 2. Filter all feature min/max values.
        # This line gets the feature names that must be filtered.
        feat2filter = []
        for k in newkeys:
            # k[:-4] because we want to crop " min" and " max"
            if k[:-4] in dfn.scalar_feature_names:
                feat2filter.append(k[:-4])

        for f in force:
            # Manually add forced features
            if f in dfn.scalar_feature_names:
                feat2filter.append(f)
            else:
                # Make sure the feature name is valid.
                raise ValueError("Unknown feature name {}".format(f))

        feat2filter = np.unique(feat2filter)

        for feat in feat2filter:
            if feat in rtdc_ds:
                fstart = feat + " min"
                fend = feat + " max"
                # Get the current feature filter
                feat_filt = self[feat]
                feat_filt[:] = True
                # If min and max exist and if they are not identical:
                if (fstart in cfg_cur and
                    fend in cfg_cur and
                        cfg_cur[fstart] != cfg_cur[fend]):
                    # TODO: speedup
                    # Here one could check for smaller values in the
                    # lists oldvals/newvals that we defined above.
                    # Be sure to check against `force` in that case!
                    ivalstart = cfg_cur[fstart]
                    ivalend = cfg_cur[fend]
                    if ivalstart > ivalend:
                        msg = "inverting filter: {} > {}".format(fstart, fend)
                        warnings.warn(msg)
                        ivalstart, ivalend = ivalend, ivalstart
                    data = rtdc_ds[feat]
                    # treat nan-values in a special way
                    disnan = np.isnan(data)
                    if np.sum(disnan):
                        # this avoids RuntimeWarnings (invalid value
                        # encountered due to nan-values)
                        feat_filt[disnan] = False
                        idx = ~disnan
                        if not cfg_cur["remove invalid events"]:
                            msg = "Feature {} contains ".format(feat) \
                                  + "nan-values! Box filters remove those."
                            warnings.warn(msg, NanWarning)
                    else:
                        idx = slice(0, len(self.all))  # place-holder for [:]
                    feat_filt[idx] &= ivalstart <= data[idx]
                    feat_filt[idx] &= data[idx] <= ivalend

        # 3. Filter with polygon filters
        # Before proceeding with the polygon filters (which are
        # computationally expensive), we compute what we have
        # so far and only compute polygon filters for those points.
        allwop = self.invalid & self.manual
        for feat in self._filters:
            allwop &= self._filters[feat]

        # check if something has changed
        pf_id = "polygon filters"
        if ((pf_id in cfg_cur and pf_id not in cfg_old)
            or (pf_id in cfg_cur and pf_id in cfg_old
                and cfg_cur[pf_id] != cfg_old[pf_id])):
            self.polygon = allwop.copy()
            # perform polygon filtering
            for p in PolygonFilter.instances:
                if p.unique_id in cfg_cur[pf_id]:
                    # update self.polygon only for so-far unfiltered data
                    # (this is faster)
                    datax = rtdc_ds[p.axes[0]][allwop]
                    datay = rtdc_ds[p.axes[1]][allwop]
                    self.polygon[allwop] &= p.filter(datax, datay)

        # 4. Finally combine `allwop` with the polygon filter
        # get a list of all filters
        if cfg_cur["enable filters"]:
            self.all[:] = allwop & self.polygon
            # Filter with configuration keyword argument "limit events".
            # This additional step limits the total number of events in
            # self.all.
            if cfg_cur["limit events"] > 0:
                limit = cfg_cur["limit events"]
                sub = self.all[self.all]
                _f, idx = downsampling.downsample_rand(sub,
                                                       samples=limit,
                                                       ret_idx=True)
                sub[~idx] = False
                self.all[self.all] = sub

        # Actual filtering is then done during plotting
        self._old_config = rtdc_ds.config.copy()["filtering"]
