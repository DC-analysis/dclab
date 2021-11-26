# flake8: noqa: F401
import warnings

from .feat_const import (
    FEATURES_SCALAR, FEATURES_NON_SCALAR, FLUOR_TRACES,
    feature_names, feature_labels, feature_name2label, scalar_feature_names
    )
from .feat_logic import (
    check_feature_shape, feature_exists, get_feature_label,
    scalar_feature_exists
    )
from .meta_const import (
    CFG_ANALYSIS, CFG_METADATA,
    # these should not be used
    config_descr, config_funcs, config_keys, config_types
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
