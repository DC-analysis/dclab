import warnings


class MachineLearningFeature:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "`MachineLearningFeature` has been stripped from dclab since "
            "version 0.61.0.")


def load_ml_feature(*args, **kwargs):
    raise NotImplementedError(
        "`load_ml_feature` has been stripped from dclab since "
        "version 0.61.0.")


def load_modc(*args, **kwargs):
    raise NotImplementedError(
        "`load_modc` has been stripped from dclab since "
        "version 0.61.0.")


def remove_all_ml_features():
    warnings.warn("The `remove_all_ml_features` does nothing since it was "
                  "stripped from dclab version 0.61.0.",
                  DeprecationWarning)


def save_modc(*args, **kwargs):
    raise NotImplementedError(
        "`save_modc` has been stripped from dclab since "
        "version 0.61.0.")
