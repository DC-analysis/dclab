
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


def compute_ml_class(mm):
    """Compute the most-probable class membership for all events

    This method also checks whether

    - there are any nan values in the ml_score features
    - the ml_score features are in the interval [0, 1]

    Notes
    -----
    I initially thought about also checking whether each feature
    sums to one, but discarded the idea. Assume that a classifier
    does a really bad classification and classifies all events in
    the same way. If the dataset is cropped at some point (e.g.
    debris or other events), then this bad classifier has an
    increased probability compared to another classifier which is
    really good at picking out one population. The ml_score values
    should be just in the range of [0, 1]. This also simplifies
    export to hdf5 and the work with hierarchy children.
    """
    feats = get_ml_score_names(mm)
    # first check each feature for nans and boundaries
    for ft in feats:
        if np.sum(np.isnan(mm[ft])):
            raise ValueError("Feature '{}' has nan values!".format(ft))
        elif np.sum(mm[ft] > 1):
            raise ValueError("Feature '{}' has values > 1!".format(ft))
        elif np.sum(mm[ft] < 0):
            raise ValueError("Feature '{}' has values < 0!".format(ft))
    # now compute the maximum for each event
    scores = [mm[ft] for ft in feats]
    matrix = np.vstack(scores)
    ml_class = np.argmax(matrix, axis=0)
    return ml_class


def has_ml_scores(mm):
    """Check whether the dataset has ml_scores defined"""
    return bool(get_ml_score_names(mm))


def register():
    AncillaryFeature(feature_name="ml_class",
                     method=compute_ml_class,
                     req_func=has_ml_scores,
                     )
