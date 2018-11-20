#!/usr/bin/python
# -*- coding: utf-8 -*-
"""RT-DC dictionary format"""
from __future__ import division, print_function, unicode_literals

import time

from .config import Configuration
from .core import RTDCBase
from .util import hashobj


class RTDC_Dict(RTDCBase):
    def __init__(self, ddict, *args, **kwargs):
        """Dictionary-based RT-DC dataset

        Parameters
        ----------
        ddict: dict
            Dictionary with keys from `dclab.definitions.feature_names`
            (e.g. "area_cvx", "deform", "image") with which the class
            will be instantiated. The configuration is set to the
            default configuration of dclab.
        *args:
            Arguments for `RTDCBase`
        **kwargs:
            Keyword arguments for `RTDCBase`
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
        self.title = "{}_{:02d}_{:02d}/{}.dict".format(t[0], t[1], t[2], ids)

        # Populate events
        self._events = {}
        for key in ddict:
            self._events[key] = ddict[key]

        event_count = len(ddict[list(ddict.keys())[0]])

        self.config = Configuration()
        self.config["experiment"]["event count"] = event_count
        # Set up filtering
        self._init_filters()

    @property
    def hash(self):
        return self._ids
