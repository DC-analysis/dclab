"""
.. versionadded:: 0.38.0
"""
import hashlib

from ..feat_anc_core import AncillaryFeature
from ...util import obj2bytes

from . import modc


class MachineLearningFeature(AncillaryFeature):
    def __init__(self, feature_name, dc_model, modc_path=None):
        """A user-defined machine-learning feature

        Parameters
        ----------
        feature_name: str
            name of the ML feature score (starts with `ml_score_`)
        dc_model: dclab.rtdc_dataset.feat_anc_ml.ml_model.BaseModel
            ML model to register
        modc_path: str or Path
            path to the original .modc file (if applicable)

        Notes
        -----
        `MachineLearningFeature` inherits from :class:`AncillaryFeature
        <dclab.rtdc_dataset.feat_anc_core.ancillary_feature.AncillaryFeature>`.
        """
        if (not feature_name.startswith("ml_score_")
                or len(feature_name) != len("ml_score_123")):
            raise ValueError("Feature name for MachineLearning Feature must be"
                             + "in the form 'ml_score_xyz',"
                             + f"got '{feature_name}'!")
        # Make sure this MachineLearningFeature has not already been
        # registered (for normal features this is ok, but here we want
        # to avoid any possible ambiguity).
        for af in AncillaryFeature.features:
            if (isinstance(af, MachineLearningFeature)
                    and af.feature_name == feature_name):
                raise ValueError("Cannot register two MachineLearningFeatures "
                                 + f"for the same feature '{feature_name}'!")

        self.feature_name = feature_name
        self.modc_path = modc_path
        self.ml_feature_info = self._process_ml_info(dc_model.info)

        # Instantiate the super class
        super(MachineLearningFeature, self).__init__(
            feature_name=feature_name,
            method=dc_model.predict,
            req_features=dc_model.inputs,
            data=dc_model,
            identifier=self.ml_feature_info["identifier"],
        )

    def _process_ml_info(self, original_info):
        """Return dictionary with all relevant info for this instance
        """
        fidx = original_info["output features"].index(self.feature_name)

        # determine feature label
        if ("output labels" in original_info
                and original_info["output labels"][fidx]):
            label = original_info["output labels"][fidx]
        else:
            label = f"ML feature {self.feature_name}"

        md5hasher = hashlib.md5()
        md5hasher.update(original_info.get("sha256", "").encode())
        md5hasher.update(original_info.get("description", "").encode())
        md5hasher.update(original_info.get("long description", "").encode())
        md5hasher.update(original_info.get("date", "").encode())
        md5hasher.update(self.feature_name.encode("utf-8"))
        for feat in original_info["input features"]:
            md5hasher.update(feat.encode("utf-8"))
        identifier = md5hasher.hexdigest()

        feature_info = {
            "date": original_info.get("date", "no date"),
            "description": original_info.get(
                "description", "No description provided"),
            "long description": original_info.get(
                "long description", "No long description provided."),
            "feature name": self.feature_name,
            "feature label": label,
            "features required": original_info["input features"],
            "scalar feature": True,
            "modc path": self.modc_path,
            "identifier": identifier,
        }

        return feature_info

    def hash(self, rtdc_ds):
        """Used for identifying an ancillary ML computation

        The required features, the used configuration keys/values, and
        the return value of the requirement function are hashed. In
        addition, the ML model's info dictionary is hashed.
        """
        super_hash = super(MachineLearningFeature, self).hash(rtdc_ds)
        hasher = hashlib.md5()
        hasher.update(obj2bytes(super_hash))
        hasher.update(obj2bytes(self.data.info))
        return hasher.hexdigest()


def load_ml_feature(modc_path):
    """Find and load MachineLearningFeature(s) from a .modc file

    Parameters
    ----------
    modc_path: str or Path
        pathname to a .modc file

    Returns
    -------
    ml_list: list of MachineLearningFeature
        list of MachineLearningFeature instances loaded from `modc_path`

    See Also
    --------
    MachineLearningFeature: class handling the plugin feature information
    """
    dc_models = modc.load_modc(modc_path)

    mlf_list = []
    for dc_model in dc_models:
        for feat in dc_model.outputs:
            mlf_list.append(MachineLearningFeature(feat, dc_model, modc_path))

    return mlf_list


def remove_all_ml_features():
    """Convenience function for removing all `MachineLearningFeature` instances

    See Also
    --------
    remove_ml_feature: remove a single `MachineLearningFeature` instance
    """
    for ml_instance in reversed(MachineLearningFeature.features):
        if isinstance(ml_instance, MachineLearningFeature):
            remove_ml_feature(ml_instance)


def remove_ml_feature(ml_instance):
    """Convenience function for removing a `MachineLearningFeature` instance

    Parameters
    ----------
    ml_instance: MachineLearningFeature
        The `MachineLearningFeature` instance to be removed from dclab

    Raises
    ------
    TypeError
        If the `ml_instance` is not a `MachineLearningFeature` instance
    """
    if isinstance(ml_instance, MachineLearningFeature):
        MachineLearningFeature.feature_names.remove(ml_instance.feature_name)
        MachineLearningFeature.features.remove(ml_instance)
    else:
        raise TypeError(f"Type {type(ml_instance)} should be an instance "
                        f"of MachineLearningFeature; got '{ml_instance}'!")
