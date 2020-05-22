#!/usr/bin/python
# -*- coding: utf-8 -*-
"""RT-DC dictionary format"""
from __future__ import division, print_function, unicode_literals

import time

import numpy as np

from .. import definitions as dfn
from ..util import hashobj

from .config import Configuration
from .core import RTDCBase


class RTDC_Dict(RTDCBase):
    def __init__(self, ddict, *args, **kwargs):
        """Dictionary-based RT-DC dataset

        Parameters
        ----------
        ddict: dict
            Dictionary with features as keys (valid features like
            "area_cvx", "deform", "image" are defined by
            `dclab.definitions.feature_exists`) with which the class
            will be instantiated. The configuration is set to the
            default configuration of dclab.

            .. versionchanged:: 0.27.0
                Scalar features are automatically converted to arrays.
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
            if dfn.feature_exists(key):
                if dfn.scalar_feature_exists(key):
                    data = np.array(ddict[key])
                else:
                    data = ddict[key]
            else:
                raise ValueError("Invalid feature name '{}'".format(key))
            self._events[key] = data

        event_count = len(ddict[list(ddict.keys())[0]])

        self.config = Configuration()
        self.config["experiment"]["event count"] = event_count
        # Set up filtering
        self._init_filters()

    @property
    def hash(self):
        return self._ids
