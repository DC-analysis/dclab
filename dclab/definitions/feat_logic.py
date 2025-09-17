import re

from . import feat_const


ML_SCORE_REGEX = re.compile(r"^ml_score_[a-z0-9]{3}$")


def check_feature_shape(name, data):
    """Check if (non)-scalar feature matches with its data's dimensionality

    Parameters
    ----------
    name: str
        name of the feature
    data: array-like
        data whose dimensionality will be checked

    Raises
    ------
    ValueError
        If the data's shape does not match its scalar description
    """
    if len(data.shape) == 1 and not scalar_feature_exists(name):
        raise ValueError(
            f"Feature '{name}' is not a scalar feature, but "
            "a 1D array was given for `data`!"
        )
    elif len(data.shape) != 1 and scalar_feature_exists(name):
        raise ValueError(
            f"Feature '{name}' is a scalar feature, but the "
            "`data` array is not 1D!"
        )


def feature_exists(name, scalar_only=False):
    """Return True if `name` is a valid feature name

    This function not only checks whether `name` is in
    :const:`feature_names`, but also validates against
    the machine learning scores `ml_score_???` (where
    `?` can be a digit or a lower-case letter in the
    English alphabet).

    Parameters
    ----------
    name: str
        name of a feature
    scalar_only : bool
        Specify whether the check should only search in scalar features

    Returns
    -------
    valid: bool
        True if name is a valid feature, False otherwise.

    See Also
    --------
    scalar_feature_exists: Wraps `feature_exists` with `scalar_only=True`
    """
    valid = False
    if name in feat_const.scalar_feature_names:
        # scalar feature
        valid = True
    elif not scalar_only and name in feat_const.feature_names:
        # non-scalar feature
        valid = True
    elif ML_SCORE_REGEX.match(name):
        # machine-learning score feature ml_score_???
        valid = True
    return valid


def feature_register(name, label=None, is_scalar=True):
    """Register a new feature for usage in dclab

    Used by temporary features and plugin features to add new feature
    names and labels to `dclab.definitions`.

    Parameters
    ----------
    name: str
        name of a feature
    label: str, optional
        feature label corresponding to the feature name. If set to None, then
        a label is constructed for the feature name.
    is_scalar: bool
        Specify whether the feature of an event is a scalar (True)
        or not (False)

    Raises
    ------
    ValueError
        If the feature already exists.
    """
    allowed_chars = "abcdefghijklmnopqrstuvwxyz_1234567890"
    feat = "".join([f for f in name if f in allowed_chars])
    if feat != name:
        raise ValueError(
            "`feature` must only contain lower-case characters, "
            f"digits, and underscores; got '{name}'!"
        )
    if label is None:
        label = f"User-defined feature {name}"
    if feature_exists(name):
        raise ValueError(f"Feature '{name}' already exists!")

    # Populate the new feature in all dictionaries and lists
    # (we don't need global here)
    feat_const.feature_names.append(name)
    feat_const.feature_labels.append(label)
    feat_const.feature_name2label[name] = label
    if is_scalar:
        feat_const.scalar_feature_names.append(name)


def feature_deregister(name):
    """Unregister a feature from dclab

    Used by temporary features and plugin features to
    remove the feature names and labels from `dclab.definitions`.

    Parameters
    ----------
    name: str
        name of a feature

    Warnings
    --------
    This function should only be used internally, i.e., You should not use
    this function. This function can break things.
    """
    label = get_feature_label(name)
    feat_const.feature_names.remove(name)
    feat_const.feature_labels.remove(label)
    feat_const.feature_name2label.pop(name)
    if name in feat_const.scalar_feature_names:
        feat_const.scalar_feature_names.remove(name)


def get_feature_label(name, rtdc_ds=None, with_unit=True):
    """Return the label corresponding to a feature name

    This function not only checks :const:`feature_name2label`,
    but also supports registered `ml_score_???` features.

    Parameters
    ----------
    name: str
        name of a feature
    with_unit: bool
        set to False to remove units in square brackets

    Returns
    -------
    label: str
        feature label corresponding to the feature name

    Notes
    -----
    TODO: extract feature label from ancillary information when an rtdc_ds is
    given.
    """
    if name in feat_const.feature_name2label:
        label = feat_const.feature_name2label[name]
    elif ML_SCORE_REGEX.match(name):
        # use a generic name for machine-learning features
        label = f"ML score {name[-3:].upper()}"
    else:
        exists = feature_exists(name)
        msg = f"Could not find label for '{name}'"
        msg += " (feature does not exist)" if not exists else ""
        raise ValueError(msg)
    if not with_unit:
        if label.endswith("]") and label.count("["):
            label = label.rsplit("[", 1)[0].strip()
    return label


def scalar_feature_exists(name):
    """Convenience method wrapping `feature_exists(..., scalar_only=True)`"""
    return feature_exists(name, scalar_only=True)
