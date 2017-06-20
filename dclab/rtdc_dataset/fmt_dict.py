#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
RT-DC dictionary file format
"""
from __future__ import division, print_function, unicode_literals

import time

import numpy as np

from dclab import definitions as dfn
from .config import Configuration
from .core import RTDCBase


class RTDC_Dict(RTDCBase):
    def __init__(self, ddict):
        """
        Parameters
        ----------
        ddict: dict
            Dictionary with keys from `dclab.definitions.uid` (e.g. "area", "defo")
            with which the class will be instantiated.
            The configuration is set to the default configuration of dclab.
        
        Notes
        -----
        Besides the filter arrays for each data column, there is a manual
        boolean filter array ``RTDCBase._filter_manual`` that can be edited
        by the user - a boolean value of ``False`` means that the event is 
        excluded from all computations.
        """
        super(RTDC_Dict, self).__init__()

        t = time.localtime()
        rand = "".join([ hex(r)[2:-1] for r in np.random.randint(10000,
                                                                 size=3)])
        self.title = "{}_{:02d}_{:02d}/{}.dict".format(t[0],t[1],t[2],rand)
        self.identifier = rand
        self.name = rand
        self.path = "none"
        self.fdir = "none"

        self._events = {}
        for key in ddict:
            kk = dfn.cfgmaprev[key.lower()]
            self._events[kk] = ddict[key]

        fill0 = np.zeros(len(ddict[list(ddict.keys())[0]]))
        for key in dfn.rdv:
            if not key in self._events:
                self._events[key] = fill0

        # Set up filtering
        self.config = Configuration(rtdc_ds=self)
        self._init_filters()
