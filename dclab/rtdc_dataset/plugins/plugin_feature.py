
# Create PlugInFeature class which is child of AncillaryFeature
# it should have methods that can be overridden by a script
# The script should create a child class of PlugInFeature
# The methods defined in the script class will override the
# PlugInFeature class.

import pathlib
import importlib
import sys

from ..ancillaries import AncillaryFeature


def create_new_plugin_feature(plugin_path):
    """find an instanciate a PlugInFeature from a user-defined script"""
    info = find_plugin_feature_script(plugin_path)

    for ii in range(len(info["feature names"])):
        ref_info = {
            "feature_name": info["feature names"][ii],
            "method": info["method"],
            "req_config": info["config required"],
            "req_features": info["features required"],
            "priority": info["priority"],
        }
        PlugInFeature(**ref_info)


def find_plugin_feature_script(plugin_path):
    # find script, return info dict
    path = pathlib.Path(plugin_path)
    # insert the plugin directory to sys.path so we can import it
    sys.path.insert(-1, str(path.parent))
    plugin = importlib.import_module(path.stem)
    # undo our path insertion
    sys.path.pop(0)
    return plugin.info


class PlugInFeature(AncillaryFeature):
    def __init__(self, **kwargs):
        """Child class of `AncillaryFeature` which allows a user to create
        their own features. See the dclab repo examples/plugins folder for
        example plugins.
        """
        super().__init__(**kwargs)
        self.plugin_registered = True
        self.plugin_info = kwargs
