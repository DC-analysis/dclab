"""
.. versionadded:: 0.34.0
"""

import pathlib
import importlib
import sys

from ... import definitions as dfn
from ..ancillaries import AncillaryFeature


class PluginImportError(BaseException):
    pass


def load_plugin_feature(plugin_path):
    """find an instanciate a PlugInFeature from a user-defined script"""
    info = import_plugin_feature_script(plugin_path)
    if not isinstance(info["feature names"], list):
        raise ValueError(
            "'feature names' must be a list of strings.")

    plugin_list = []
    for feature_name in info["feature names"]:
        plugin_list.append(PlugInFeature(feature_name, info, plugin_path))
    return plugin_list


def import_plugin_feature_script(plugin_path):
    # find script, return info dict
    path = pathlib.Path(plugin_path)
    try:
        # insert the plugin directory to sys.path so we can import it
        sys.path.insert(-1, str(path.parent))
        plugin = importlib.import_module(path.stem)
    except ModuleNotFoundError:
        raise PluginImportError("The plugin could be not be found at "
                                f"'{plugin_path}'!")
    finally:
        # undo our path insertion
        sys.path.pop(0)

    return plugin.info


def remove_plugin_feature(plugin_instance):
    """Convenience function for removing a plugin instance"""
    if isinstance(plugin_instance, PlugInFeature):
        # This check is necessary for situations where the PlugInFeature fails
        # between updating the `dclab.dfn` file and initialising the
        # AncillaryFeature
        if plugin_instance.feature_name in PlugInFeature.feature_names:
            PlugInFeature.feature_names.remove(plugin_instance.feature_name)
            dfn._remove_feature_from_definitions(plugin_instance.feature_name)
        PlugInFeature.features.remove(plugin_instance)
    else:
        raise TypeError(f"Type {type(plugin_instance)} should be an instance "
                        f"of PlugInFeature. '{plugin_instance}' was given.")


def remove_all_plugin_features():
    """Convenience function for removing all plugin instances"""
    for plugin_instance in reversed(PlugInFeature.features):
        if isinstance(plugin_instance, PlugInFeature):
            remove_plugin_feature(plugin_instance)


class PlugInFeature(AncillaryFeature):
    def __init__(self, feature_name, info, plugin_path=None):
        """Child class of `AncillaryFeature` which allows a user to create
        their own features. See the dclab repo examples/plugins folder for
        example plugins.
        """
        self._plugin_feature_name = feature_name
        self._plugin_original_info = info
        self.plugin_path = plugin_path
        self.plugin_feature_info = self._handle_plugin_info()
        self._handle_ancill_info()
        super().__init__(**self._ancill_info)

    def _handle_plugin_info(self):
        """Grab the relevant plugin feature from `info` and then set the
        default `info` values if necessary.
        """
        self._error_check_plugin_original_info()
        _label, _is_scalar = self._update_feature_name_and_label()
        plugin_feature_info = {
            "method": self._plugin_original_info["method"],
            "description": self._plugin_original_info.get(
                "description", "Description of my feature"),
            "long description": self._plugin_original_info.get(
                "long description", "Long description of my feature"),
            "feature name": self._plugin_feature_name,
            "feature label": _label,
            "features required": self._plugin_original_info.get(
                "features required", []),
            "config required": self._plugin_original_info.get(
                "config required", []),
            "method check required": self._plugin_original_info.get(
                "method check required", lambda x: True),
            "scalar feature": _is_scalar,
            "version": self._plugin_original_info.get("version", "unknown"),
        }
        return plugin_feature_info

    def _handle_ancill_info(self):
        self._ancill_info = {
            "feature_name": self.plugin_feature_info["feature name"],
            "method": self.plugin_feature_info["method"],
            "req_config": self.plugin_feature_info["config required"],
            "req_features": self.plugin_feature_info["features required"],
            "req_func": self.plugin_feature_info["method check required"],
        }

    def _update_feature_name_and_label(self):
        """Wrapper on the `dclab.dfn._add_feature_to_definitions` function for
        handling cases when `feature_label=""`

        This should be excecuted before initializing the super class
        (AncillaryFeature). If we don't do this, then `remove_plugin_feature`
        may end up removing innate features e.g., "deform".

        """

        idx = self._plugin_original_info["feature names"].index(
            self._plugin_feature_name)
        _is_scalar = self._plugin_original_info["scalar feature"][idx]
        _label = self._plugin_original_info["feature labels"][idx]
        if self.feature_label == "":
            self.feature_label = None
        dfn._add_feature_to_definitions(
            self._plugin_feature_name, _label, _is_scalar)
        if self.feature_label is None:
            self.feature_label = dfn.get_feature_label(
                self._plugin_feature_name)
        return _label, _is_scalar

    def _error_check_plugin_original_info(self):
        if not isinstance(self._plugin_original_info, dict):
            raise ValueError(
                "PlugInFeature input for `info` must be a dict, instead a "
                f"'{type(self._plugin_original_info)}' was given.")

        if self._plugin_feature_name not in \
                self._plugin_original_info["feature names"]:
            raise ValueError(
                "PlugInFeature input for `feature_name` was not found in the "
                "input `info` dict. `feature_name` = '{}', "
                "`info['feature names']` = '{}'. ".format(
                    self._plugin_feature_name,
                    self._plugin_original_info['feature names']))

        if "feature labels" not in self._plugin_original_info:
            raise ValueError(
                "'feature labels' was not found in the input `info` dict. ")

        if not callable(self._plugin_original_info["method"]):
            raise ValueError(
                "The `method` you have provided in the input `info` is not "
                f"callable ('{self._plugin_original_info['method']}' is not "
                "a function).")
