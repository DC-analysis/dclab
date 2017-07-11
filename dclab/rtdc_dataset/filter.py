#!/usr/bin/python
# -*- coding: utf-8 -*-
"""RT-DC dataset core classes and methods"""
from __future__ import division, print_function, unicode_literals


import numpy as np

from dclab import definitions as dfn


from .. import downsampling
from ..polygon_filter import PolygonFilter



class Filter(object):
    def __init__(self, rtdc_ds):
        """Boolean filter arrays for RT-DC measurements
        
        Parameters
        ----------
        rtdc_ds: instance of RTDCBase
            The RT-DC data set the filter applies to
        
        
        """
        self.rtdc_ds = rtdc_ds
        self._filters = {}

        # All filters combined (see `self.update`)
        self.all = np.ones(len(rtdc_ds), dtype=bool)
        # invalid (nan/inf) events
        self.invalid = np.ones(len(rtdc_ds), dtype=bool)
        # reserved for filtering
        self.manual = np.ones(len(rtdc_ds), dtype=bool)
        # polygon filters
        self.polygon = np.ones(len(rtdc_ds), dtype=bool)
        # old filter configuration of `rtdc_ds`
        self._old_config = {}


    def __getitem__(self, key):
        """Return the filter for a column of `self.rtdc_ds`"""
        if key in self.rtdc_ds:
            if (key not in self._filters and
                key in dfn.column_names):
                # Generate filters on-the-fly
                self._filters[key] = np.ones(len(self.rtdc_ds), dtype=bool)
        return self._filters[key]


    def update(self, force=[]):
        """Update the filters according to `self.rtdc_ds.config["filtering"]`


        Parameters
        ----------
        force : list
            A list of column names that must be refiltered with
            min/max values.
        """

        # These lists may help us become very fast in the future
        newkeys = []
        oldvals = []
        newvals = []

        cfg_cur = self.rtdc_ds.config["filtering"]
        cfg_old = self._old_config

        ## Determine which data was updated        
        for skey in list(cfg_cur.keys()):
            if not skey in cfg_old:
                cfg_old[skey] = None
            if cfg_cur[skey] != cfg_old[skey]:
                newkeys.append(skey)
                oldvals.append(cfg_old[skey])
                newvals.append(cfg_cur[skey])

        # 1. Filter all column min/max values.
        # This line gets the column names that must be filtered.
        col2filter = []
        for k in newkeys:
            # k[:-4] because we want to crop " min" and " max"
            if k[:-4] in dfn.column_names:
                col2filter.append(k[:-4])


        for f in force:
            # Manually add forced columns
            if f in dfn.column_names:
                col2filter.append(f)
            else:
                # Make sure the column name is valid.
                raise ValueError("Unknown column name {}".format(f))
        
        col2filter = np.unique(col2filter)

        for col in col2filter:
            if col in self.rtdc_ds:
                fstart = col + " min"
                fend = col + " max"
                # Get the current column filter
                col_filt = self[col]
                # If min and max exist and if they are not identical:
                if (fstart in cfg_cur and
                    fend in cfg_cur and
                    cfg_cur[fstart] != cfg_cur[fend]):
                    # TODO: speedup
                    # Here one could check for smaller values in the
                    # lists oldvals/newvals that we defined above.
                    # Be sure to check against force in that case!
                    ivalstart = cfg_cur[fstart]
                    ivalend = cfg_cur[fend]
                    data = self.rtdc_ds[col]
                    col_filt[:] = (ivalstart <= data)*(data <= ivalend)
                else:
                    col_filt[:] = True
        
        # 2. Filter with polygon filters
        # check if something has changed
        pf_id = "polygon filters"
        if (
            (pf_id in cfg_cur and not pf_id in cfg_old) or
            (pf_id in cfg_cur and pf_id in cfg_old and
             cfg_cur[pf_id] != cfg_old[pf_id])):
            self.polygon[:] = True
            # perform polygon filtering
            for p in PolygonFilter.instances:
                if p.unique_id in cfg_cur[pf_id]:
                    # update self.polygon
                    # iterate through axes
                    datax = self.rtdc_ds[p.axes[0]]
                    datay = self.rtdc_ds[p.axes[1]]
                    self.polygon *= p.filter(datax, datay)

        # 3. Invalid filters
        self.invalid[:] = True
        if cfg_cur["remove invalid events"]:            
            for col in dfn.column_names:
                if col in self.rtdc_ds:
                    data = self.rtdc_ds[col]
                    invalid = np.isinf(data)+np.isnan(data)
                    self.invalid *= ~invalid

        # 4. Finally combine all filters
        # get a list of all filters
        self.all[:] = True

        if cfg_cur["enable filters"]:
            for col in self._filters:
                self.all[:] *= self._filters[col]
            
            self.all[:] *= self.invalid
            self.all[:] *= self.manual
            self.all[:] *= self.polygon
    
            # Filter with configuration keyword argument "Limit Events".
            # This additional step limits the total number of events in
            # self.all.
            if cfg_cur["limit events"] > 0:
                limit = cfg_cur["limit events"]
                sub = self.all[self.all]
                _f, idx = downsampling.downsample_rand(sub,
                                                       samples=limit,
                                                       retidx=True)
                sub[~idx] = False
                self.all[self.all] = sub

        
        # Actual filtering is then done during plotting            
        self._old_config = self.rtdc_ds.config.copy()["filtering"]
