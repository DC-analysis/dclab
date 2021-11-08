import collections

from ..ml_libs import import_or_mock_package
from ..ml_model import BaseModel


tf = import_or_mock_package("tensorflow", "2.0")


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
        tf.keras.models.save_model(model=bare_model,
                                   save_format=save_format,
                                   filepath=path)

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
