"""
.. versionadded:: 0.34.0
"""

import pathlib
import importlib
import sys

from ..ancillaries import AncillaryFeature


_registered_plugin_instances = []


def create_new_plugin_feature(plugin_path):
    """find an instanciate a PlugInFeature from a user-defined script"""
    info = find_plugin_feature_script(plugin_path)
    plugin_list = []
    for ii in range(len(info["feature names"])):
        ref_info = {
            "feature_name": info["feature names"][ii],
            "method": info["method"],
            "req_config": info["config required"],
            "req_features": info["features required"],
            "priority": info["priority"],
        }
        plugin_list.append(PlugInFeature(**ref_info))
        # add feature label etc
    return plugin_list


def find_plugin_feature_script(plugin_path):
    # find script, return info dict
    path = pathlib.Path(plugin_path)
    # insert the plugin directory to sys.path so we can import it
    sys.path.insert(-1, str(path.parent))
    plugin = importlib.import_module(path.stem)
    # undo our path insertion
    sys.path.pop(0)
    return plugin.info


def remove_plugin_feature(plugin_instance):
    """Convenience function for removing a plugin instance"""
    PlugInFeature.features.remove(plugin_instance)
    PlugInFeature.feature_names.remove(plugin_instance.feature_name)


def remove_all_plugin_features():
    # I guess because we are built on AncillaryFeature, we don't need something
    # like temp feats' _registered_temporary_features = []?
    # I have done this below anyway...
    for plugin in _registered_plugin_instances:
        remove_plugin_feature(plugin)


class PlugInFeature(AncillaryFeature):
    def __init__(self, **kwargs):
        """Child class of `AncillaryFeature` which allows a user to create
        their own features. See the dclab repo examples/plugins folder for
        example plugins.
        """
        super().__init__(**kwargs)
        self.plugin_registered = True
        self.plugin_info = kwargs
