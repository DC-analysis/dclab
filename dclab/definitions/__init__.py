# flake8: noqa: F401
import warnings

from .feat_const import (
    FEATURES_SCALAR, FEATURES_NON_SCALAR, FLUOR_TRACES,
    # these should not be used
    feature_names, feature_labels, feature_name2label,
    # this one should also not be used, but we wait with deprecation,
    # because Shape-Out heavily relies on it (it shouldn't)
    scalar_feature_names
    )
from .feat_logic import (
    check_feature_shape, feature_exists, get_feature_label,
    scalar_feature_exists
    )
from .meta_const import (
    CFG_ANALYSIS, CFG_METADATA, config_keys,
    # these should not be used (in contrast, `config_keys` is quite
    # important for iterating over all internal configuration keys)
    config_descr, config_funcs, config_types,
    )
from .meta_logic import (
    config_key_exists,
    get_config_value_descr, get_config_value_func, get_config_value_type
    )


class DeprecatedObject:
    def __init__(self, anobject, name, new):
        self.anobject = anobject
        self.message = f"dclab.dfn.{name} is deprecated, please use " \
            + f"dclab.dfn.{new} instead."

    def __getattr__(self, item):
        warnings.warn(self.message, DeprecationWarning)
        return self.anobject.__getattr__(item)

    def __getitem__(self, item):
        warnings.warn(self.message, DeprecationWarning)
        return self.anobject.__getitem__(item)

    def __iter__(self):
        warnings.warn(self.message, DeprecationWarning)
        return self.anobject.__iter__()

    def __contains__(self, item):
        warnings.warn(self.message, DeprecationWarning)
        return self.anobject.__contains__(item)


config_descr = DeprecatedObject(config_descr,
                                "config_descr",
                                "get_config_value_descr",
                                )

config_funcs = DeprecatedObject(config_funcs,
                                "config_funcs",
                                "get_config_value_func",
                                )

config_types = DeprecatedObject(config_types,
                                "config_types",
                                "get_config_value_type",
                                )

feature_names = DeprecatedObject(feature_names,
                                 "feature_names",
                                 "feature_exists",
                                 )

feature_labels = DeprecatedObject(feature_labels,
                                  "feature_labels",
                                  "get_feature_label",
                                  )

feature_name2label = DeprecatedObject(feature_name2label,
                                      "feature_name2label",
                                      "get_feature_label",
                                      )
