#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Computation of ancillary features

Ancillary features are computed on-the-fly in dclab if the
required data is available. The features are registered here
and are computed when `RTDCBase.__getitem__` is called with
the respective feature name. When `RTDCBase.__contains__` is
called with the feature name, then the feature is not yet
computed, but the prerequisites are evaluated:

In [1]: "emodulus" in rtdc_dataset  # nothing is computed
Out[1]: True
In [2]: rtdc_dataset["emodulus"]
Out[2]: ndarray([...])  # now data is computed and cached

Once the data has been computed, `RTDCBase` caches it in
the `_ancillaries` property dict together with a hash
that is computed with `AncillaryFeature.hash`. The hash
is computed from the feature data `req_features` and the
configuration metadata `req_config`.
"""
from __future__ import division, print_function, unicode_literals

import hashlib
import warnings

import numpy as np

from ..util import obj2str


class BadFeatureSizeWarning(UserWarning):
    pass


class AncillaryFeature():
    # Holds all instances of this class
    features = []
    feature_names = []

    def __init__(self, feature_name, method, req_config=[], req_features=[],
                 priority=0):
        """A data feature that is computed from existing data

        Parameters
        ----------
        feature_name: str
            The name of the ancillary feature, e.g. "emodulus".
        method: callable
            The method that computes the feature. This method
            takes an instance of `RTDCBase` as argument.
        req_config: list
            Required configuration parameters to compute the feature,
            e.g. ["calculation", ["emodulus model", "emodulus viscosity"]]
        req_features: list
            Required existing features in the dataset,
            e.g. ["area_cvx", "deform"]
        priority: int
            The priority of the feature; if there are multiple
            AncillaryFeature defined for the same feature_name,
            then the priority of the features defines which feature
            returns True in `self.is_available`. A higher value
            means a higher priority.

        Notes
        -----
        `req_config` and `req_features` are used to test whether the
        feature can be computed in `self.is_available`.
        """
        self.feature_name = feature_name
        self.method = method
        self.req_config = req_config
        self.req_features = req_features
        self.priority = priority

        # register this feature
        AncillaryFeature.features.append(self)
        AncillaryFeature.feature_names.append(feature_name)

    def __repr__(self):
        return "Ancillary feature: {}".format(self.feature_name)

    @staticmethod
    def available_features(rtdc_ds):
        """Determine available features for an RT-DC dataset

        Parameters
        ----------
        rtdc_ds: instance of RTDCBase
            The dataset to check availability for

        Returns
        -------
        features: dict
            Dictionary with feature names as keys and instances
            of `AncillaryFeature` as values.
        """
        cols = {}
        for inst in AncillaryFeature.features:
            if inst.is_available(rtdc_ds):
                cols[inst.feature_name] = inst
        return cols

    def compute(self, rtdc_ds):
        """Compute the feature with self.method

        Parameters
        ----------
        rtdc_ds: instance of RTDCBase
            The dataset to compute the feature for

        Returns
        -------
        feature: array- or list-like
            The computed data feature (read-only).
        """
        data = self.method(rtdc_ds)
        dsize = len(rtdc_ds) - len(data)

        if dsize > 0:
            msg = "Growing feature {} in {} by {} to match event number!"
            warnings.warn(msg.format(self.feature_name, rtdc_ds, abs(dsize)),
                          BadFeatureSizeWarning)
            data.resize(len(rtdc_ds), refcheck=False)
            data[-dsize:] = np.nan
        elif dsize < 0:
            msg = "Shrinking feature {} in {} by {} to match event number!"
            warnings.warn(msg.format(self.feature_name, rtdc_ds, abs(dsize)),
                          BadFeatureSizeWarning)
            data.resize(len(rtdc_ds), refcheck=False)

        if isinstance(data, np.ndarray):
            data.setflags(write=False)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, np.ndarray):
                    item.setflags(write=False)

        return data

    @staticmethod
    def get_instances(feature_name):
        """Return all all instances that compute `feature_name`"""
        feats = []
        for ft in AncillaryFeature.features:
            if ft.feature_name == feature_name:
                feats.append(ft)
        return feats

    def hash(self, rtdc_ds):
        """Used for identifying an ancillary computation

        The data columns and the used configuration keys/values
        are hashed.
        """
        hasher = hashlib.md5()
        # data columns
        for col in self.req_features:
            hasher.update(obj2str(rtdc_ds[col]))
        # config keys
        for sec, keys in self.req_config:
            for key in keys:
                val = rtdc_ds.config[sec][key]
                data = "{}:{}={}".format(sec, key, val)
                hasher.update(obj2str(data))
        return hasher.hexdigest()

    def is_available(self, rtdc_ds, verbose=False):
        """Check whether the feature is available

        Parameters
        ----------
        rtdc_ds: instance of RTDCBase
            The dataset to check availability for

        Returns
        -------
        available: bool
            `True`, if feature can be computed with `compute`

        Notes
        -----
        This method returns `False` for a feature if there
        is a feature defined with the same name but with
        higher priority (even if the feature would be
        available otherwise).
        """
        # Check config keys
        for item in self.req_config:
            section, keys = item
            if section not in rtdc_ds.config:
                if verbose:
                    print("{} not in config".format(section))
                return False
            else:
                for key in keys:
                    if key not in rtdc_ds.config[section]:
                        if verbose:
                            print("{} not in config['{}']".format(key,
                                                                  section))
                        return False
        # Check features
        for col in self.req_features:
            if col not in rtdc_ds:
                return False
        # Check priorities of other features
        for of in AncillaryFeature.features:
            if of == self:
                # nothing to compare
                continue
            elif of.feature_name == self.feature_name:
                # same feature name
                if of.priority <= self.priority:
                    # lower priority, ignore
                    continue
                else:
                    # higher priority
                    if of.is_available(rtdc_ds):
                        # higher priority is available, thus
                        # this feature is not available
                        return False
                    else:
                        # higher priority not available
                        continue
            else:
                # other feature
                continue
        return True
