"""DCOR feature handling"""
from functools import lru_cache
import numbers

import numpy as np

from ... import definitions as dfn


class DCORNonScalarFeature:
    """Helper class for accessing non-scalar features"""

    def __init__(self, feat, api, size):
        self.identifier = api.url + ":" + feat  # for caching ancillaries
        self.feat = feat
        self.api = api
        self._size = size

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, event):
        if not isinstance(event, numbers.Integral):
            # slicing!
            indices = np.arange(len(self))[event]
            trace0 = self._get_item(indices[0])
            # determine the correct shape from the first feature
            oshape = [len(indices)] + list(trace0.shape)
            output = np.zeros(oshape, dtype=trace0.dtype)
            # populate the output array
            for ii, evid in enumerate(indices):
                output[ii] = self._get_item(evid)
            return output
        else:
            return self._get_item(event)

    def __len__(self):
        return self._size

    @lru_cache(maxsize=100)
    def _get_item(self, event):
        data = self.api.get(query="feature", feat=self.feat, event=event)
        return np.asarray(data)


class DCORContourFeature(DCORNonScalarFeature):
    """Helper class for accessing contour data"""

    def __init__(self, feat, api, size):
        super(DCORContourFeature, self).__init__(feat, api, size)
        self.shape = (size, np.nan, 2)

    def __getitem__(self, event):
        if not isinstance(event, numbers.Integral):
            # We cannot use the original method, because contours
            # may have different sizes! So we return a list.
            indices = np.arange(len(self))[event]
            output = []
            # populate the output list
            for evid in indices:
                output.append(self._get_item(evid))
            return output
        else:
            return self._get_item(event)


class DCORImageFeature(DCORNonScalarFeature):
    """Helper class for accessing image data"""

    def __init__(self, feat, api, size):
        super(DCORImageFeature, self).__init__(feat, api, size)
        metadata = self.api.get(query="metadata")
        self.shape = (size,
                      metadata["imaging"]["roi size y"],
                      metadata["imaging"]["roi size x"])


class DCORTraceItem(DCORNonScalarFeature):
    """Helper class for accessing traces"""
    def __init__(self, feat, api, size, samples_per_event):
        super(DCORTraceItem, self).__init__(feat, api, size)
        self.shape = (size, samples_per_event)

    @lru_cache(maxsize=100)
    def _get_item(self, event):
        data = self.api.get(query="feature", feat="trace",
                            trace=self.feat, event=event)
        return np.asarray(data)


class DCORTraceFeature:
    """Helper class for accessing traces"""

    def __init__(self, api, size):
        self.identifier = api.url + ":traces"
        self.api = api
        self._size = size
        metadata = self.api.get(query="metadata")
        self._samples_per_event = metadata["fluorescence"]["samples per event"]
        self.traces = api.get(query="trace_list")
        self._trace_objects = {}

        self.shape = (len(self.traces),
                      size,
                      self._samples_per_event
                      )

    def __contains__(self, key):
        return key in self.traces

    def __getitem__(self, trace):
        if trace in self.traces:
            if trace not in self._trace_objects:
                self._trace_objects[trace] = DCORTraceItem(
                    trace, self.api, self._size, self._samples_per_event)
            return self._trace_objects[trace]
        else:
            raise KeyError(f"trace '{trace}' not found!")

    def __len__(self):
        return len(self.traces)

    def keys(self):
        return self.traces


class FeatureCache:
    """Download and cache (scalar only) features from DCOR"""

    def __init__(self, api, size):
        self.api = api
        self._features = self.api.get(query="feature_list")
        self._size = size
        self._scalar_cache = {}
        self._nonsc_features = {}

    def __contains__(self, key):
        return key in self._features

    def __getitem__(self, key):
        # user-level checking is done in core.py
        assert dfn.feature_exists(key)
        if key not in self._features:
            raise KeyError(f"Feature '{key}' not found!")

        if key in self._scalar_cache:
            return self._scalar_cache[key]
        elif dfn.scalar_feature_exists(key):
            # download the feature and cache it
            feat = np.asarray(self.api.get(query="feature", feat=key))
            self._scalar_cache[key] = feat
            return feat
        elif key == "contour":
            if key not in self._nonsc_features:
                self._nonsc_features[key] = DCORContourFeature(key, self.api,
                                                               self._size)
            return self._nonsc_features[key]
        elif key == "trace":
            if "trace" not in self._nonsc_features:
                self._nonsc_features["trace"] = DCORTraceFeature(self.api,
                                                                 self._size)
            return self._nonsc_features["trace"]
        elif key in ["image", "mask"]:
            self._nonsc_features[key] = DCORImageFeature(key, self.api,
                                                         self._size)
            return self._nonsc_features[key]
        else:
            raise ValueError(f"No DCOR handler for feature '{key}'!")

    def __iter__(self):
        # dict-like behavior
        for key in self.keys():
            yield key

    def keys(self):
        return self._features
