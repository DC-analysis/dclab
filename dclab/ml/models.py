import abc
import collections
import uuid

import numpy as np

from .mllibs import tensorflow as tf


class BaseModel(abc.ABC):
    def __init__(self, model, inputs, outputs, model_name=None,
                 output_names=None):
        self.model = model
        self.inputs = inputs
        self.outputs = outputs
        self.name = model_name or str(uuid.uuid4())[:5]
        self.output_names = output_names or inputs

    @abc.abstractmethod
    def predict(self, ds, *args, **kwargs):
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
        pass

    def get_dataset_features(self, ds, dtype=np.float32):
        """Return the dataset features used for inference

        Parameters
        ----------
        ds: dclab.rtdc_dataset.RTDCBase
            Dataset from which to retrieve the feature data
        dtype: np.dtype
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
        # TODO
        pass

    def unregister(self):
        """Unregister from dclab ancillary features"""
        # TODO
        pass


class TensorflowModel(BaseModel):
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
        To obtain the probabilities, `tf.keras.layers.Softmax()`
        is used.
        """
        probability_model = tf.keras.Sequential([self.model,
                                                 tf.keras.layers.Softmax()])
        fdata = self.get_dataset_features(ds)
        tfdata = tf.data.Dataset.from_tensor_slices(fdata).batch(batch_size)
        ret = probability_model.predict(tfdata)
        ofdict = collections.OrderedDict()
        for ii, key in enumerate(self.outputs):
            ofdict[key] = ret[:, ii]
        return ofdict
