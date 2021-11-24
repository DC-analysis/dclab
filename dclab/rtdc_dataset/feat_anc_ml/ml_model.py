import abc

import numpy as np


class BaseModel(abc.ABC):
    def __init__(self, bare_model, inputs, outputs, info=None):
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
        info: dict
            Dictionary with model metadata
        """
        if info is None:
            info = {}
        info.setdefault("input features", inputs)
        info.setdefault("output features", outputs)
        self.bare_model = bare_model
        self.inputs = inputs
        self.outputs = outputs
        self.info = info
        self.output_labels = info.get("output labels") or outputs

    @staticmethod
    def all_formats():
        """Dict of dictionaries containing all model formats in dclab

        Returns
        -------
        fmt_dict: dict
            All file formats with names as keys.
            Each item contains the keys "name" (format name),
            "suffix" (saved file suffix), "requires" (Python
            dependencies).

        See Also
        --------
        supported_formats: class-specific file formats
        """
        formats = {}
        for cls in BaseModel.__subclasses__():
            # Register the tensorflow file format
            for fmt in cls.supported_formats():
                formats[fmt["name"]] = {
                    "requirements": fmt["requirements"],
                    "suffix": fmt["suffix"],
                    "class": cls}
        return formats

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

        Notes
        -----
        The return value is automatically added to the return value
        of :func:`BaseModel.all_formats`.
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
