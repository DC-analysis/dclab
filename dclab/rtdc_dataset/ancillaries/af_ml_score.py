from .ancillary_feature import AncillaryFeature


def register(dc_model):
    """Register an ML model

    For user convenience, this function can also be invoked by calling
    `dc_model.register`. Models can be removed from the registration
    by calling `unregister` or `dc_model.unregister`.

    Parameters
    ----------
    dc_model: dclab.ml.models.BaseModel
        ML model to register
    """
    # Check whether the model is already registered
    for af in list(AncillaryFeature.features):
        if af.data is dc_model:
            return
    # Check for feature existence
    for feat in dc_model.outputs:
        # Do not allow registration of the same feature at once
        for of in AncillaryFeature.features:
            if of.feature_name == feat:
                raise ValueError("Feature '{}' is already ".format(feat)
                                 + "defined by another ancillary feature "
                                 + "with the model '{}'.".format(of.data))

    # Register ml_score_??? features
    for feat in dc_model.outputs:
        # create an ancillary feature for each output
        def method(ds, feature=feat):
            return dc_model.predict(ds)[feature]
        AncillaryFeature(feature_name=feat,
                         method=method,
                         req_config=[],
                         req_features=dc_model.inputs,
                         data=dc_model,
                         )


def unregister(dc_model):
    """Unregister an ML model

    See :func:`register` for more information.
    """
    for af in list(AncillaryFeature.features):
        if af.data is dc_model:
            AncillaryFeature.features.remove(af)
            # Do not break here! There are multiple AncillaryFeatures
            # for one model.
