"""
.. versionadded:: 0.34.0
"""
import hashlib
import importlib
import pathlib
import sys

from ...definitions import feat_logic
from ..feat_anc_core import AncillaryFeature


class PluginImportError(BaseException):
    pass


def import_plugin_feature_script(plugin_path):
    """Import the user-defined recipe and return the info dictionary

    Parameters
    ----------
    plugin_path: str or Path
        pathname to a valid dclab plugin script

    Returns
    -------
    info: dict
        Dictionary with the information required to instantiate
        one (or multiple) :class:`PlugInFeature`.

    Raises
    ------
    PluginImportError
        If the plugin can not be found

    Notes
    -----
    One recipe may define multiple plugin features.
    """
    path = pathlib.Path(plugin_path)
    if not path.exists():
        raise PluginImportError("The plugin could be not be found at "
                                f"'{plugin_path}'!")
    try:
        # insert the plugin directory to sys.path so we can import it
        sys.path.insert(-1, str(path.parent))
        sys.dont_write_bytecode = True
        plugin = importlib.import_module(path.stem)
    except BaseException as e:
        raise PluginImportError(
            f"The plugin {plugin_path} could not be loaded!") from e
    finally:
        # undo our path insertion
        sys.path.pop(0)
        sys.dont_write_bytecode = False

    return plugin.info


def load_plugin_feature(plugin_path):
    """Find and load PlugInFeature(s) from a user-defined recipe

    Parameters
    ----------
    plugin_path: str or Path
        pathname to a valid dclab plugin Python script

    Returns
    -------
    plugin_list: list of PlugInFeature
        list of PlugInFeature instances loaded from `plugin_path`

    Raises
    ------
    ValueError
        If the script dictionary "feature names" are not a list

    Notes
    -----
    One recipe may define multiple plugin features.

    See Also
    --------
    import_plugin_feature_script: function that imports the plugin script
    PlugInFeature: class handling the plugin feature information
    dclab.rtdc_dataset.feat_temp.register_temporary_feature: alternative
        method for creating user-defined features
    """
    info = import_plugin_feature_script(plugin_path)
    if not isinstance(info["feature names"], list):
        raise ValueError(
            "'feature names' must be a list of strings.")

    plugin_list = []
    for feature_name in info["feature names"]:
        plugin_list.append(PlugInFeature(feature_name, info, plugin_path))
    return plugin_list


def remove_all_plugin_features():
    """Convenience function for removing all `PlugInFeature` instances

    See Also
    --------
    remove_plugin_feature: remove a single `PlugInFeature` instance
    """
    for plugin_instance in reversed(PlugInFeature.features):
        if isinstance(plugin_instance, PlugInFeature):
            remove_plugin_feature(plugin_instance)


def remove_plugin_feature(plugin_instance):
    """Convenience function for removing a `PlugInFeature` instance

    Parameters
    ----------
    plugin_instance: PlugInFeature
        The `PlugInFeature` instance to be removed from dclab

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
            feat_logic.feature_deregister(plugin_instance.feature_name)
        PlugInFeature.features.remove(plugin_instance)
    else:
        raise TypeError(f"Type {type(plugin_instance)} should be an instance "
                        f"of PlugInFeature. '{plugin_instance}' was given.")


class PlugInFeature(AncillaryFeature):
    def __init__(self, feature_name, info, plugin_path=None):
        """A user-defined plugin feature

        Parameters
        ----------
        feature_name: str
            name of a feature that matches that defined in `info`
        info: dict
            Full plugin recipe (for all features) as given in the
            `info` dictionary in the plugin file.
            At least the following keys must be specified:

            - "method": callable function computing the plugin feature
              values (takes an :class`dclab.rtdc_dataset.core.RTDCBase`
              as argument)
            - "feature names": list of plugin feature names provided
              by the plugin

            The following features are optional:

            - "description": short (one-line) description of the plugin
            - "long description": long description of the plugin
            - "feature labels": feature labels used e.g. for plotting
            - "feature shapes": list of tuples for each feature
              indicating the shape (this is required only for
              non-scalar features; for scalar features simply set
              this to ``None`` or ``(1,)``).
            - "scalar feature": list of boolean values indicating
              whether the features are scalar
            - "config required": configuration keys required to
              compute the plugin features (see the `req_config`
              parameter for :class:`.AncillaryFeature`)
            - "features required": list of feature names required to
              compute the plugin features (see the `req_features`
              parameter for :class:`.AncillaryFeature`)
            - "method check required": additional method that checks
              whether the features can be computed (see the `req_func`
              parameter for :class:`.AncillaryFeature`)
            - "version": version of this plugin (please use
              semantic verioning)

        plugin_path: str or Path, optional
            path which was used to load the `PlugInFeature` with
            :func:`load_plugin_feature`.

        Notes
        -----
        `PluginFeature` inherits from :class:`AncillaryFeature
        <dclab.rtdc_dataset.feat_anc_core.ancillary_feature.AncillaryFeature>`.
        Please read the advanced section on `PluginFeatures` in the dclab docs.
        """
        if plugin_path is not None:
            plugin_path = pathlib.Path(plugin_path)

        #: Plugin feature name
        self.feature_name = feature_name
        #: Path to the original plugin file
        self.plugin_path = plugin_path

        # perform sanity checks
        self._sanity_check_original_info(info)
        # keep this for tests
        self._original_info = info

        # populate `info` dictionary with missing values
        #: Dictionary containing all information relevant for
        #: this particular plugin feature instance
        self.plugin_feature_info = self._process_plugin_info(info)

        # register this plugin feature in definitions
        # This must be executed before initializing the super class
        # (AncillaryFeature). If we don't do this, then `remove_plugin_feature`
        # may end up removing innate features e.g., "deform".
        feat_logic.feature_register(
            name=self.feature_name,
            label=self.plugin_feature_info["feature label"],
            is_scalar=self.plugin_feature_info["scalar feature"]
        )

        # Instantiate the super class
        super(PlugInFeature, self).__init__(
            feature_name=self.plugin_feature_info["feature name"],
            method=self.plugin_feature_info["method"],
            req_config=self.plugin_feature_info["config required"],
            req_features=self.plugin_feature_info["features required"],
            req_func=self.plugin_feature_info["method check required"],
            identifier=self.plugin_feature_info["identifier"],
        )

    def _process_plugin_info(self, original_info):
        """Return dictionary with all relevant info for this instance
        """
        fidx = original_info["feature names"].index(self.feature_name)

        # determine feature label
        if ("feature labels" in original_info
                and original_info["feature labels"][fidx]):
            label = original_info["feature labels"][fidx]
        else:
            label = f"Plugin feature {self.feature_name}"

        # determine whether we have a scalar feature
        if "scalar feature" in original_info:
            is_scalar = original_info["scalar feature"][fidx]
        else:
            is_scalar = True  # default

        if is_scalar:
            event_shape = (1,)
        else:
            if "feature shapes" in original_info:
                event_shape = original_info["feature shapes"][fidx]
            else:
                event_shape = None

        # We assume that the script does not import any other custom
        # Python scripts.
        md5hasher = hashlib.md5()
        if self.plugin_path is not None:
            md5hasher.update(self.plugin_path.read_bytes())
        else:
            md5hasher.update(original_info["method"].__code__.co_code)
        md5hasher.update(self.feature_name.encode("utf-8"))
        md5hasher.update(original_info.get("version", "").encode("utf-8"))
        for feat in original_info.get("features required", []):
            md5hasher.update(feat.encode("utf-8"))
        identifier = md5hasher.hexdigest()

        feature_info = {
            "method": original_info["method"],
            "description": original_info.get(
                "description", "No description provided"),
            "long description": original_info.get(
                "long description", "No long description provided."),
            "feature name": self.feature_name,
            "feature label": label,
            "feature shape": event_shape,
            "features required": original_info.get("features required", []),
            "config required": original_info.get("config required", []),
            "method check required": original_info.get(
                "method check required", lambda x: True),
            "scalar feature": is_scalar,
            # allow comparisons with distutil.version.LooseVersion
            "version": original_info.get("version", "0.0.0-unknown"),
            "plugin path": self.plugin_path,
            "identifier": identifier,
        }

        return feature_info

    def _sanity_check_original_info(self, original_info):
        """Various checks on the `original_info` attribute dict

        Raises
        ------
        ValueError
            If the parameter `original_info` is not a dict.
            If the `self.feature_name` is not in
                `original_info["feature names"]`.
            If the `method` provided in parameter `original_info`
                is not callable.
        """
        if not isinstance(original_info, dict):
            raise ValueError(
                "PlugInFeature parameter for `info` must be a dict, instead "
                f"a '{type(original_info)}' was given.")

        if not isinstance(original_info["feature names"], list):
            raise ValueError("'feature names' must be a list, "
                             f"got '{type(original_info['feature names'])}'")

        if self.feature_name not in original_info["feature names"]:
            raise ValueError(
                f"The feature name '{self.feature_name}' is not defined in "
                + "the `info` dict of the plugin feature"
                + (f" at {self.plugin_path}" if self.plugin_path else "")
                + f". Defined names are '{original_info['feature names']}'.")

        if not callable(original_info["method"]):
            raise ValueError(
                "The `method` you have provided in the parameter `info` is "
                f"not callable ('{original_info['method']}' is not "
                "a function).")
