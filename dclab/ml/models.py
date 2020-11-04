import abc
import collections
import uuid

import numpy as np

from .mllibs import tensorflow as tf
from ..rtdc_dataset.ancillaries import af_ml_score


class BaseModel(abc.ABC):
    def __init__(self, bare_model, inputs, outputs, model_name=None,
                 output_labels=None):
        """
        Parameters
        ----------
        bare_model:
            Underlying ML model
        inputs: list of str
            List of model input features, e.g.
            ``["deform", "area_um"]``
        outputs: list of str
            List of output features the model provides in that order, e.g.
            ``["ml_score_rbc", "ml_score_rt1", "ml_score_tfe"]``
        model_name: str or None
            The name of the models
        output_labels: list of str
            List of more descriptive labels for the features, e.g.
            ``["red blood cell", "type 1 cell", "troll cell"]``.
        """
        self.bare_model = bare_model
        self.inputs = inputs
        self.outputs = outputs
        self.name = model_name or str(uuid.uuid4())[:5]
        self.output_labels = output_labels or outputs

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *args):
        self.unregister()

    @staticmethod
    @abc.abstractmethod
    def supported_formats():
        """List of dictionaries containing model formats

        Returns
        -------
        fmts: list
            Each item contains the keys "name" (format name),
            "suffix" (saved file suffix), "requires" (Python
            dependencies).
        """

    @staticmethod
    @abc.abstractmethod
    def load_bare_model(path):
        """Load an implementation-specific model from a file

        This will set the `self.model` attribute. Make sure that
        the other attributes are set properly as well.
        """

    @staticmethod
    @abc.abstractmethod
    def save_bare_model(path, bare_model, save_format=None):
        """Save an implementation-specific model to a file

        Parameters
        ----------
        path: str or path-like
            Path to store model to
        bare_model: object
            The implementation-specific bare model
        save_format: str
            Must be in `supported_formats`
        """

    @abc.abstractmethod
    def predict(self, ds):
        """Return the probabilities of `self.outputs` for `ds`

        Parameters
        ----------
        ds: dclab.rtdc_dataset.RTDCBase
            Dataset to apply the model to

        Returns
        -------
        ofdict: dict
            Output feature dictionary with features as keys
            and 1d ndarrays as values.

        Notes
        -----
        This function calls :func:`BaseModel.get_dataset_features`
        to obtain the input feature matrix.
        """

    def get_dataset_features(self, ds, dtype=np.float32):
        """Return the dataset features used for inference

        Parameters
        ----------
        ds: dclab.rtdc_dataset.RTDCBase
            Dataset from which to retrieve the feature data
        dtype: dtype
            All features are cast to this dtype

        Returns
        -------
        fdata: 2d ndarray
            2D array of shape (len(ds), len(self.inputs));
            i.e. to access the array containing the first feature,
            for all events, you would do `fdata[:, 0]`.
        """
        fdata = np.zeros((len(ds), len(self.inputs)), dtype=dtype)
        for ii, feat in enumerate(self.inputs):
            fdata[:, ii] = ds[feat]
        return fdata

    def register(self):
        """Register this model to the dclab ancillary features"""
        af_ml_score.register(self)

    def unregister(self):
        """Unregister from dclab ancillary features"""
        af_ml_score.unregister(self)


class TensorflowModel(BaseModel):
    """Handle tensorflow models"""
    @staticmethod
    def supported_formats():
        return [{"name": "tensorflow-SavedModel",
                 "suffix": ".tf",
                 "requirements": "tensorflow"}
                ]

    @staticmethod
    def load_bare_model(path):
        """Load a tensorflow model"""
        # We don't use tf.saved_model.load, because it does not
        # return a keras layer.
        bare_model = tf.keras.models.load_model(str(path))
        return bare_model

    @staticmethod
    def save_bare_model(path, bare_model, save_format="tensorflow-SavedModel"):
        """Save a tensorflow model"""
        assert save_format == "tensorflow-SavedModel"
        tf.saved_model.save(obj=bare_model, export_dir=str(path))

    def predict(self, ds, batch_size=32):
        """Return the probabilities of `self.outputs` for `ds`

        Parameters
        ----------
        ds: dclab.rtdc_dataset.RTDCBase
            Dataset to apply the model to
        batch_size: int
            Batch size for inference with tensorflow

        Returns
        -------
        ofdict: dict
            Output feature dictionary with features as keys
            and 1d ndarrays as values.

        Notes
        -----
        Before prediction, this method asserts that the outputs of the
        model are converted to probabilities. If the final layer
        is one-dimensional and does not have a sigmoid activation,
        then a sigmoid activation layer is added (binary
        classification) ``tf.keras.layers.Activation("sigmoid")``.
        If the final layer has more dimensions and is not a
        ``tf.keras.layers.Softmax()`` layer, then a softmax layer
        is added.
        """
        probability_model = tf.keras.Sequential([self.bare_model])
        if self.bare_model.output_shape[1] > 1:
            # Multiple outputs; check for softmax
            if not self.has_softmax_layer():
                probability_model.add(tf.keras.layers.Softmax())
        else:
            # Binary classification; check for sigmoid
            if not self.has_sigmoid_activation():
                probability_model.add(tf.keras.layers.Activation("sigmoid"))

        fdata = self.get_dataset_features(ds)
        tfdata = tf.data.Dataset.from_tensor_slices(fdata).batch(batch_size)
        ret = probability_model.predict(tfdata)
        ofdict = collections.OrderedDict()
        for ii, key in enumerate(self.outputs):
            ofdict[key] = ret[:, ii]
        return ofdict

    def has_sigmoid_activation(self, layer_config=None):
        """Return True if final layer has "sigmoid" activation function"""
        if layer_config is None:
            layer_config = self.bare_model.get_config()
        if "layers" in layer_config:
            return self.has_sigmoid_activation(layer_config["layers"][-1])
        else:
            activation = layer_config.get("config", "").get("activation", "")
            return activation == "sigmoid"

    def has_softmax_layer(self, layer_config=None):
        """Return True if final layer is a Softmax layer"""
        if layer_config is None:
            layer_config = self.bare_model.get_config()
        if "layers" in layer_config:
            return self.has_softmax_layer(layer_config["layers"][-1])
        else:
            return layer_config["class_name"] == "Softmax"
