"""
.. versionadded:: 0.33.0
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..definitions import feat_logic

from .core import RTDCBase
from .fmt_hierarchy import RTDC_Hierarchy, map_indices_child2root


_registered_temporary_features = []


def deregister_all():
    """Deregisters all temporary features"""
    for feat in list(_registered_temporary_features):
        deregister_temporary_feature(feat)


def deregister_temporary_feature(feature: str):
    """Convenience function for deregistering a temporary feature

    This method is mostly used during testing. It does not
    remove the actual feature data from any dataset; the data
    will stay in memory but is not accessible anymore through
    the public methods of the :class:`RTDCBase` user interface.
    """
    if feature in _registered_temporary_features:
        _registered_temporary_features.remove(feature)
        feat_logic.feature_deregister(feature)


def register_temporary_feature(feature: str,
                               label: Optional[str] = None,
                               is_scalar: bool = True):
    """Register a new temporary feature

    Temporary features are custom features that can be defined ad hoc
    by the user. Temporary features are helpful when the integral
    features are not enough, e.g. for prototyping, testing, or
    collating with other data. Temporary features allow you to
    leverage the full functionality of :class:`RTDCBase` with
    your custom features (no need to go for a custom `pandas.Dataframe`).

    Parameters
    ----------
    feature: str
        Feature name; allowed characters are lower-case letters,
        digits, and underscores
    label: str
        Feature label used e.g. for plotting
    is_scalar: bool
        Whether or not the feature is a scalar feature
    """
    feat_logic.feature_register(feature, label, is_scalar)
    _registered_temporary_features.append(feature)


def set_temporary_feature(rtdc_ds: RTDCBase,
                          feature: str,
                          data: np.ndarray):
    """Set temporary feature data for a dataset

    Parameters
    ----------
    rtdc_ds: dclab.RTDCBase
        Dataset for which to set the feature. Note that the
        length of the feature `data` must match the number of events
        in `rtdc_ds`. If the dataset is a hierarchy child, the data will also
        be set in the parent dataset, but only for those events that are part
        of the child. For all events in the parent dataset that are not part
        of the child dataset, the temporary feature is set to np.nan.
    feature: str
        Feature name
    data: np.ndarray
        The data
    """
    if not feat_logic.feature_exists(feature):
        raise ValueError(
            f"Temporary feature '{feature}' has not been registered!")
    if len(data) != len(rtdc_ds):
        raise ValueError(f"The temporary feature {feature} must have same "
                         f"length as the dataset. Expected length "
                         f"{len(rtdc_ds)}, got length {len(data)}!")
    if isinstance(rtdc_ds, RTDC_Hierarchy):
        root_ids = map_indices_child2root(rtdc_ds, np.arange(len(rtdc_ds)))
        root_parent = rtdc_ds.get_root_parent()
        root_feat_data = np.empty((len(root_parent)))
        root_feat_data[:] = np.nan
        root_feat_data[root_ids] = data
        set_temporary_feature(root_parent, feature, root_feat_data)
        rtdc_ds.rejuvenate()
    else:
        feat_logic.check_feature_shape(feature, data)
        rtdc_ds._usertemp[feature] = data
