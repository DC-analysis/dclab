"""RT-DC dictionary format"""
import time

import numpy as np

from .. import definitions as dfn
from ..util import hashobj

from .config import Configuration
from .core import RTDCBase


class DictContourEvent:
    def __init__(self, contours):
        assert contours[0].shape[1] == 2
        self.shape = (len(contours), np.nan, 2)
        self.contours = contours

    def __iter__(self):
        return iter(self.contours)

    def __getitem__(self, item):
        return self.contours[item]

    def __len__(self):
        return len(self.contours)


class DictTraceEvent(dict):

    @property
    def shape(self):
        key0 = sorted(self.keys())[0]
        return len(self), len(self[key0]), len(self[key0][0])


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
        for feat in ddict:
            if dfn.feature_exists(feat):
                if dfn.scalar_feature_exists(feat):
                    data = np.array(ddict[feat])
                elif feat == "contour":
                    data = DictContourEvent(ddict[feat])
                elif feat == "trace":
                    data = DictTraceEvent(ddict[feat])
                else:
                    data = ddict[feat]
            else:
                raise ValueError("Invalid feature name '{}'".format(feat))
            self._events[feat] = data

        event_count = len(ddict[list(ddict.keys())[0]])

        self.config = Configuration()
        self.config["experiment"]["event count"] = event_count
        # Set up filtering
        self._init_filters()

    @property
    def hash(self):
        return self._ids
