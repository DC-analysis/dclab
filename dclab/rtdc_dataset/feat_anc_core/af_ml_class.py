
import numpy as np
from .ancillary_feature import AncillaryFeature


def get_ml_score_names(mm):
    """Return a list of all ml_score_??? features"""
    feats = []
    # We cannot loop over mm.features because of infinite recursions
    for ft in mm._feature_candidates:
        if ft.startswith("ml_score_") and ft in mm:
            feats.append(ft)
    return sorted(feats)


def compute_ml_class(mm, sanity_checks=True):
    """Compute the most-probable class membership for all events

    Parameters
    ----------
    mm: .RTDCBase
        instance with the `ml_score_???` features
    sanity_checks: bool
        set to `False` to not perform sanity checks (checks whether
        the scores are between 0 and 1)

    Returns
    -------
    ml_class: 1D ndarray
        The most-probable class for each event in `mm`. If no class
        can be attributed to an event (because the scores are all
        `np.nan` or `0` for that event), the class `-1` is used.

    Notes
    -----
    I initially thought about also checking whether each feature
    sums to one, but discarded the idea. Let's assume that a classifier
    does an awful classification and classifies all events in
    the same way. If the dataset is cropped at some point (e.g.
    debris or other events), then this bad classifier has an
    increased probability compared to another classifier which is
    perfect at picking out one population. The ml_score values
    should be just in the range of [0, 1]. This also simplifies
    export to hdf5 and the work with hierarchy children.
    """
    feats = get_ml_score_names(mm)

    # the score matrix
    score_matrix = np.zeros((len(mm), len(feats)), dtype=float)

    for ii, ft in enumerate(feats):
        if sanity_checks:
            if np.nanmax(mm[ft]) > 1:
                raise ValueError("Feature '{}' has values > 1!".format(ft))
            elif np.nanmin(mm[ft]) < 0:
                raise ValueError("Feature '{}' has values < 0!".format(ft))
        score_matrix[:, ii] = mm[ft]

    # Now compute the maximum for each event. The initial idea was to just
    # use `ml_class = np.nanargmax(score_matrix, axis=1)`. However, here we
    # run into these problems:
    # 1. This does not handle All-NaN slices, e.g. all features are `np.nan`
    #    for an event.
    # 2. This does not properly handle manually-rated, zero-valued features,
    #    e.g. in a situation where we have two features, one with `np.nan`
    #    and one with `0`, we cannot assign the event to either of the two
    #    classes.
    # 3. There is no "unclassified" class (this also becomes apparent in
    #    point 2). We will set all events that cannot be attributed to a
    #    class to `-1` in `ml_class`.

    # Define unusable entries:
    unusable = np.logical_or(np.isnan(score_matrix), (score_matrix == 0))
    where_idx_nan = np.sum(~unusable, axis=1) == 0
    score_matrix[where_idx_nan, :] = -1
    ml_class = np.nanargmax(score_matrix, axis=1)
    ml_class[where_idx_nan] = -1
    return ml_class


def has_ml_scores(mm):
    """Check whether the dataset has ml_scores defined"""
    # Return the sorted score names plus Ancillary feature hashes.
    # This will be used to determine the hash of the ml_class feature,
    # which is important in case the user replaces an ML feature
    # with a new one.
    features = get_ml_score_names(mm)
    idlist = []
    for feat in features:
        # We also hash any other AncillaryFeature that might implement
        # this ML score. But this use case is basically non-existent and
        # the performance impact is probably negligible.
        candidates = AncillaryFeature.get_instances(feat)
        idlist.append((feat, [c.hash(mm) for c in candidates]))
    return idlist


def register():
    AncillaryFeature(feature_name="ml_class",
                     method=compute_ml_class,
                     req_func=has_ml_scores,
                     )
