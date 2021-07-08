"""RT-DC dataset core classes and methods"""

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
        # dictionary of boolean array for box filters
        self._box_filters = {}
        # dictionary of (hash, boolean array) for polygon filters
        self._poly_filters = {}
        # initialize important parameters
        self._init_rtdc_ds(rtdc_ds)
        # initialize properties
        self.reset()

    def __getitem__(self, key):
        """Return the filter for a feature in `self.features`"""
        if key in self.features and dfn.scalar_feature_exists(key):
            if key not in self._box_filters:
                # Generate filters on-the-fly
                self._box_filters[key] = np.ones(self.size, dtype=bool)
        else:
            raise KeyError("Feature not available: '{}'".format(key))
        return self._box_filters[key]

    def _init_rtdc_ds(self, rtdc_ds):
        #: Available feature names
        self.features = rtdc_ds.features_scalar
        if hasattr(self, "size") and self.size != len(rtdc_ds):
            raise ValueError("Change of RTDCBase size not supported!")
        self.size = len(rtdc_ds)
        # determine box filters that have been removed
        for key in list(self._box_filters.keys()):
            if key not in self.features:
                self._box_filters.pop(key)
        # determine polygon filters that have been removed
        for pf_id in list(self._poly_filters.keys()):
            pf = PolygonFilter.get_instance_from_id(pf_id)
            if (pf_id in rtdc_ds.config["filtering"]["polygon filters"]
                and pf.axes[0] in self.features
                    and pf.axes[1] in self.features):
                pass
            else:
                # filter has been removed
                self._poly_filters.pop(pf_id)

    def reset(self):
        """Reset all filters"""
        self._box_filters = {}
        self._poly_filters = {}
        #: All filters combined (see :func:`Filter.update`);
        #: Use this property to filter the features of
        #: :class:`dclab.rtdc_dataset.RTDCBase` instances
        self.all = np.ones(self.size, dtype=bool)
        #: All box filters
        self.box = np.ones(self.size, dtype=bool)
        #: Invalid (nan/inf) events
        self.invalid = np.ones(self.size, dtype=bool)
        #: 1D boolean array for manually excluding events; `False` values
        #: are excluded.
        self.manual = np.ones(self.size, dtype=bool)
        #: Polygon filters
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
            if cfg_cur[skey] != cfg_old.get(skey, None):
                newkeys.append(skey)
                oldvals.append(cfg_old.get(skey, None))
                newvals.append(cfg_cur[skey])

        # 1. Invalid filters
        self.invalid[:] = True
        if cfg_cur["remove invalid events"]:
            for feat in self.features:
                data = rtdc_ds[feat]
                invalid = np.isinf(data) | np.isnan(data)
                self.invalid &= ~invalid

        # 2. Filter all feature min/max values.
        feat2filter = []
        for k in newkeys:
            # k[:-4] because we want to crop " min" and " max"
            if (dfn.scalar_feature_exists(k[:-4])
                    and (k.endswith(" min") or k.endswith(" max"))):
                feat2filter.append(k[:-4])

        for f in force:
            # add forced features
            if dfn.scalar_feature_exists(f):
                feat2filter.append(f)
            else:
                # Make sure the feature name is valid.
                raise ValueError("Unknown scalar feature name '{}'!".format(f))

        feat2filter = np.unique(feat2filter)

        for feat in feat2filter:
            fstart = feat + " min"
            fend = feat + " max"
            must_be_filtered = (fstart in cfg_cur
                                and fend in cfg_cur
                                and cfg_cur[fstart] != cfg_cur[fend])
            if ((fstart in cfg_cur and fend not in cfg_cur)
                    or (fstart not in cfg_cur and fend in cfg_cur)):
                # User is responsible for setting min and max values!
                raise ValueError("Box filter: Please make sure that both "
                                 "'{}' and '{}' are set!".format(fstart, fend))
            if feat in self.features:
                # Get the current feature filter
                feat_filt = self[feat]
                feat_filt[:] = True
                # If min and max exist and if they are not identical:
                if must_be_filtered:
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
                            msg = "Feature '{}' contains ".format(feat) \
                                  + "nan-values! Box filters remove those."
                            warnings.warn(msg, NanWarning)
                    else:
                        idx = slice(0, len(self.all))  # place-holder for [:]
                    feat_filt[idx] &= ivalstart <= data[idx]
                    feat_filt[idx] &= data[idx] <= ivalend
            elif must_be_filtered:
                warnings.warn("Dataset '{}' does ".format(rtdc_ds.identifier)
                              + "not contain the feature '{}'! ".format(feat)
                              + "A box filter has been ignored.")
        # store box filters
        self.box[:] = True
        for feat in self._box_filters:
            self.box &= self._box_filters[feat]

        # 3. Filter with polygon filters
        # check if something has changed
        # perform polygon filtering
        for pf_id in cfg_cur["polygon filters"]:
            pf = PolygonFilter.get_instance_from_id(pf_id)
            if (pf_id not in self._poly_filters
                    or pf.hash != self._poly_filters[pf_id][0]):
                datax = rtdc_ds[pf.axes[0]]
                datay = rtdc_ds[pf.axes[1]]
                self._poly_filters[pf_id] = (pf.hash, pf.filter(datax, datay))
        # store polygon filters
        self.polygon[:] = True
        for pf_id in self._poly_filters:
            self.polygon &= self._poly_filters[pf_id][1]

        # 4. Finally combine all filters and apply "limit events"
        # get a list of all filters
        if cfg_cur["enable filters"]:
            self.all[:] = self.invalid & self.manual & self.polygon & self.box

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
        else:
            self.all[:] = True

        # Actual filtering is then done during plotting
        self._old_config = rtdc_ds.config.copy()["filtering"]
