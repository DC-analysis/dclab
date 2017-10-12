#!/usr/bin/python
# -*- coding: utf-8 -*-
"""RT-DC dictionary format"""
from __future__ import division, print_function, unicode_literals

import time

import numpy as np

from dclab import definitions as dfn
from .config import Configuration
from .core import RTDCBase
from .util import hashobj



class RTDC_Dict(RTDCBase):
    def __init__(self, ddict, *args, **kwargs):
        """Dictionary-based RT-DC data set 
        
        Parameters
        ----------
        ddict: dict
            Dictionary with keys from `dclab.definitions.column_names`
            (e.g. "area_cvx", "deform") with which the class will be
            instantiated. The configuration is set to the default
            configuration of dclab.
        *args, **kwargs:
            (Keyword) arguments for `RTDCBase`
        """
        assert ddict
        
        super(RTDC_Dict, self).__init__(*args, **kwargs)

        t = time.localtime()
        
        # Get an identifying string
        keys = list(ddict.keys())
        keys.sort()
        ids = hashobj(ddict[keys[0]])
        self._ids = ids
        self.path = "none"
        self.title = "{}_{:02d}_{:02d}/{}.dict".format(t[0], t[1], t[2],ids)


        # Populate events
        self._events = {}
        for key in ddict:
            self._events[key] = ddict[key]

        # Populate empty columns
        event_count = len(ddict[list(ddict.keys())[0]])
        fill0 = np.zeros(event_count)
        for key in dfn.column_names:
            if not key in self._events:
                self._events[key] = fill0

        self.config = Configuration()
        self.config["experiment"]["event count"] = event_count
        # Set up filtering
        self._init_filters()


    @property
    def hash(self):
        return self._ids
