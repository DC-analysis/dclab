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
    """Find and load PlugInFeature(s) from a user-defined script

    Parameters
    ----------
    plugin_path : str or Path
        pathname to a valid dclab plugin script

    Returns
    -------
    plugin_list : list
        list of `PlugInFeature`

    Raises
    ------
    ValueError
        If the script dictionary "feature names" are not a list

    See Also
    --------
    import_plugin_feature_script : function that imports the plugin script
    PlugInFeature : class handling the plugin feature information
    dclab.register_temporary_feature : alternative method for creating
        user-defined features

    """
    info = import_plugin_feature_script(plugin_path)
    if not isinstance(info["feature names"], list):
        raise ValueError(
            "'feature names' must be a list of strings.")

    plugin_list = []
    for feature_name in info["feature names"]:
        plugin_list.append(PlugInFeature(feature_name, info, plugin_path))
    return plugin_list


def import_plugin_feature_script(plugin_path):
    """Find the user-defined script and return the info dictionary

    Parameters
    ----------
    plugin_path : str or Path
        pathname to a valid dclab plugin script

    Returns
    -------
    plugin.info : dict
        dict containing the info required to instantiate a `PlugInFeature`

    Raises
    ------
    PluginImportError
        If the plugin can not be found

    """
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
    """Convenience function for removing a `PlugInFeature` instance

    Parameters
    ----------
    plugin_instance : PlugInFeature
        The `PlugInFeature` class to be removed from dclab

    Raises
    ------
    TypeError
        If the `plugin_instance` is not a `PlugInFeature` instance

    """
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
    """Convenience function for removing all `PlugInFeature` instances

    See Also
    --------
    remove_plugin_feature : remove a single `PlugInFeature` instance

    """
    for plugin_instance in reversed(PlugInFeature.features):
        if isinstance(plugin_instance, PlugInFeature):
            remove_plugin_feature(plugin_instance)


class PlugInFeature(AncillaryFeature):
    def __init__(self, feature_name, info, plugin_path=None):
        """Class that allows a user to define a feature

        Parameters
        ----------
        feature_name : str
            name of a feature that matches that defined in `info`
        info : dict
            Necessary information to create the `PlugInFeature`.
            Minimum requirements are
                "method": callable function
                "feature names": list of names
        plugin_path : str or Path, optional
            pathname which was used to load the `PlugInFeature` with
            `load_plugin_feature`.

        Attributes
        ----------
        feature_name : str
            Attribute inherited from `AncillaryFeature`.
            See `dclab.AncillaryFeature` for other inherited attributes.
        plugin_feature_info : dict
            All relevant information pertaining to the instance of
            `PlugInFeature`.

        Notes
        -----
        Child class of `AncillaryFeature` which allows a user to create
        their own features. See the dclab repository examples/plugins folder
        for example plugins.
        """
        self._plugin_feature_name = feature_name
        self._original_info = info
        self.plugin_path = plugin_path
        self._handle_plugin_info()
        self._handle_ancill_info()
        super().__init__(**self._ancill_info)

    def _handle_plugin_info(self):
        """Create the `plugin_feature_info` attribute dict.

        Use the `_original_info` attribute to populate the
        `plugin_feature_info` attribute with all relevant information
        pertaining to the instance of `PlugInFeature`.

        """
        self._error_check_original_info()
        _label, _is_scalar = self._update_feature_name_and_label()
        plugin_feature_info = {
            "method": self._original_info["method"],
            "description": self._original_info.get(
                "description", "Description of my feature"),
            "long description": self._original_info.get(
                "long description", "Long description of my feature"),
            "feature name": self._plugin_feature_name,
            "feature label": _label,
            "features required": self._original_info.get(
                "features required", []),
            "config required": self._original_info.get(
                "config required", []),
            "method check required": self._original_info.get(
                "method check required", lambda x: True),
            "scalar feature": _is_scalar,
            "version": self._original_info.get("version", "unknown"),
        }
        self.plugin_feature_info = plugin_feature_info

    def _handle_ancill_info(self):
        """Create the private `_ancill_info` attribute dict.

        Use the `plugin_feature_info` attribute to populate the
        `_ancill_info` attribute with all relevant information required
        to initialise an `AncillaryFeature`.

        """
        self._ancill_info = {
            "feature_name": self.plugin_feature_info["feature name"],
            "method": self.plugin_feature_info["method"],
            "req_config": self.plugin_feature_info["config required"],
            "req_features": self.plugin_feature_info["features required"],
            "req_func": self.plugin_feature_info["method check required"],
        }

    def _update_feature_name_and_label(self):
        """Add feature information to `dclab.definitions`.

        Returns
        -------
        _label : str
            Feature label used to populate the `plugin_feature_info` attribute
        _is_scalar : bool
            Whether the feature is a scalar

        Notes
        -----
        This must be excecuted before initializing the super class
        (AncillaryFeature). If we don't do this, then `remove_plugin_feature`
        may end up removing innate features e.g., "deform".

        """
        idx = self._original_info["feature names"].index(
            self._plugin_feature_name)
        _is_scalar = self._original_info["scalar feature"][idx]
        if "feature labels" not in self._original_info:
            _label = None
        else:
            _label = self._original_info["feature labels"][idx]
        if _label == "":
            _label = None
        dfn._add_feature_to_definitions(
            self._plugin_feature_name, _label, _is_scalar)
        if _label is None:
            _label = dfn.get_feature_label(
                self._plugin_feature_name)
        return _label, _is_scalar

    def _error_check_original_info(self):
        """Various checks on the `_original_info` attibute dict

        Raises
        ------
        ValueError
            If the parameter `info` is not a dict.
            If the parameter `feature_name` is not in
                parameter `info["feature names"]`.
            If the method provided in parameter `info` is not callable.

        """
        if not isinstance(self._original_info, dict):
            raise ValueError(
                "PlugInFeature parameter for `info` must be a dict, instead "
                f"a '{type(self._original_info)}' was given.")

        if self._plugin_feature_name not in \
                self._original_info["feature names"]:
            raise ValueError(
                "PlugInFeature parameter for `feature_name` was not found in "
                "the parameter `info` dict. `feature_name` = '{}', "
                "`info['feature names']` = '{}'. ".format(
                    self._plugin_feature_name,
                    self._original_info['feature names']))

        if not callable(self._original_info["method"]):
            raise ValueError(
                "The `method` you have provided in the parameter `info` is "
                f"not callable ('{self._original_info['method']}' is not "
                "a function).")
