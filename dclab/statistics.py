"""Statistics computation for RT-DC dataset instances"""

import numpy as np
import traceback as tb
import warnings

from . import definitions as dfn


class BadMethodWarning(UserWarning):
    pass


class Statistics(object):
    available_methods = {}

    def __init__(self, name, method, req_feature=False):
        """A helper class for computing statistics

        All statistical methods are registered in the dictionary
        `Statistics.available_methods`.
        """
        self.method = method
        self.name = name
        self.req_feature = req_feature
        Statistics.available_methods[name] = self

    def __call__(self, **kwargs):
        data = self._get_data(kwargs)
        if len(data) == 0:
            result = np.nan
        else:
            try:
                result = self.method(data)
            except BaseException:
                exc = tb.format_exc().replace("\n", "\n    | ")
                warnings.warn("Failed to compute {} for {}: {}".format(
                              self.name, kwargs["ds"].title, exc),
                              BadMethodWarning)
                result = np.nan
        return result

    def _get_data(self, kwargs):
        """Convenience wrapper to get statistics data"""
        if "ds" not in kwargs:
            raise ValueError("Keyword argument 'ds' missing.")

        ds = kwargs["ds"]

        if self.req_feature:
            if "feature" not in kwargs:
                raise ValueError("Keyword argument 'feature' missing.")
            return self.get_feature(ds, kwargs["feature"])
        else:
            return ds

    def get_feature(self, ds, feat):
        """Return filtered feature data

        The features are filtered according to the user-defined filters,
        using the information in `ds.filter.all`. In addition, all
        `nan` and `inf` values are purged.

        Parameters
        ----------
        ds: dclab.rtdc_dataset.RTDCBase
            The dataset containing the feature
        feat: str
            The name of the feature; must be a scalar feature
        """
        if ds.config["filtering"]["enable filters"]:
            x = ds[feat][ds.filter.all]
        else:
            x = ds[feat]
        bad = np.isnan(x) | np.isinf(x)
        xout = x[~bad]
        return xout


def flow_rate(ds):
    """Return the flow rate of an RT-DC dataset"""
    conf = ds.config["setup"]
    if "flow rate" in conf:
        return conf["flow rate"]
    else:
        return np.nan


def get_statistics(ds, methods=None, features=None):
    """Compute statistics for an RT-DC dataset

    Parameters
    ----------
    ds: dclab.rtdc_dataset.RTDCBase
        The dataset for which to compute the statistics.
    methods: list of str or None
        The methods wih which to compute the statistics.
        The list of available methods is given with
        `dclab.statistics.Statistics.available_methods.keys()`
        If set to `None`, statistics for all methods are computed.
    features: list of str
        Feature name identifiers are defined by
        `dclab.definitions.feature_exists`.
        If set to `None`, statistics for all scalar features
        available are computed.

    Returns
    -------
    header: list of str
        The header (feature + method names) of the computed statistics.
    values: list of float
        The computed statistics.
    """
    if methods is None:
        cls = list(Statistics.available_methods.keys())
        # sort the features in a usable way
        avm = Statistics.available_methods
        me1 = [m for m in cls if not avm[m].req_feature]
        me2 = [m for m in cls if avm[m].req_feature]
        methods = me1 + me2

    if features is None:
        features = ds.features_scalar
    else:
        features = [a.lower() for a in features]

    header = []
    values = []

    # To make sure that all methods are computed for each feature in a block,
    # we loop over all features. It would be easier to loop over the methods,
    # but the resulting statistics would not be human-friendly.
    for ft in features:
        for mt in methods:
            meth = Statistics.available_methods[mt]
            if meth.req_feature:
                if ft in ds:
                    values.append(meth(ds=ds, feature=ft))
                else:
                    values.append(np.nan)
                label = dfn.get_feature_label(ft, rtdc_ds=ds)
                header.append(" ".join([mt, label]))
            else:
                # Prevent multiple entries of this method.
                if not header.count(mt):
                    values.append(meth(ds=ds))
                    header.append(mt)

    return header, values


def mode(data):
    """Compute an intelligent value for the mode

    The most common value in experimental is not very useful if there
    are a lot of digits after the comma. This method approaches this
    issue by rounding to bin size that is determined by the
    Freedman–Diaconis rule.

    Parameters
    ----------
    data: 1d ndarray
        The data for which the mode should be computed.

    Returns
    -------
    mode: float
        The mode computed with the Freedman-Diaconis rule.
    """
    # size
    n = data.shape[0]
    # interquartile range
    iqr = np.percentile(data, 75)-np.percentile(data, 25)
    # Freedman–Diaconis
    bin_size = 2 * iqr / n**(1/3)

    if bin_size == 0:
        return np.nan

    # Add bin_size/2, because we want the center of the bin and
    # not the left corner of the bin.
    databin = np.round(data/bin_size)*bin_size + bin_size/2
    u, indices = np.unique(databin, return_inverse=True)
    mode = u[np.argmax(np.bincount(indices))]

    return mode


# Register all the methods
# Methods that require an axis
Statistics(name="Mean",   req_feature=True, method=np.average)
Statistics(name="Median", req_feature=True, method=np.median)
Statistics(name="Mode",   req_feature=True, method=mode)
Statistics(name="SD",     req_feature=True, method=np.std)
# Methods that work on RTDCBase
Statistics(name="Events",
           method=lambda mm: np.sum(mm.filter.all))
Statistics(name="%-gated",
           method=lambda mm: np.average(mm.filter.all)*100)
Statistics(name="Flow rate",
           method=lambda mm: flow_rate(mm))
