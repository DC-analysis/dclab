.. _sec_av_ml:

================
Machine learning
================

To simplify machine-learning (ML) tasks in the context of RT-DC, dclab offers
a few convenience methods. This section describes the recommended way
of implementing and distributing ML models based on RT-DC data. Please
make sure that you have installed dclab with the *ml* extra
(``pip install dclab[ml]``).


.. _sec_av_ml_models:

Using models in dclab
=====================

For RT-DC analysis, the most common task for ML is to determine the probability
for a specific event (e.g. a cell) to belong to a specific class (e.g. red
blood cell). Since RT-DC data always has a very specific format, it is
worthwile to standardize this regression/classification process.

In dclab, you are not directly using the *bare* models that you would e.g.
get from tensorflow/keras. Instead, models are wrapped via a specific
:class:`dclab.ml.models.BaseModel` class that holds additional information
about the features from which and to which a model maps. For instance,
a model might have the inputs ``deform`` and ``area_um`` and make
predictions regarding a defined output feature, e.g. ``ml_score_rbc``.
Output features for machine learning are always of the form ``ml_score_xxx``
where ``x`` can be any alphanumeric character (you are free to choose).

.. code:: python

    import dclab.ml
    import tensorflow as tf

    # do your magic
    bare_model = tf.keras.Sequential(...)
    bare_model.compile(...)
    bare_model.fit(...)

    # create a dclab model
    dc_model = dclab.ml.models.TensorflowModel(
        bare_model=bare_model,
        inputs=["deform", "area_um"],
        outputs=["ml_score_rbc"],
        model_name="RBC identification",
        output_labels=["Red Blood Cells"])

    # once you get here, you can use your model directly for inference
    ds = dclab.new_dataset("path/to/a/dataset")
    # `prediction` is a dictionary with the key "ml_score_rbc" mapping
    # to a 1D ndarray of length `len(ds)`, holding the probability data.
    prediction = dc_model.predict(ds)["ml_score_rbc"]

For user convenience, a model can also be registered with dclab as
an :ref:`ancillary feature <sec_features_ancillary>`.

.. code:: python

    dc_model.register()
    prediction = ds["ml_score_rbc"]  # same result as above
    dc_model.unregister()  # optional

If it is inconvenient for you to call the ``register()`` and ``unregister``
methods (e.g. when you would like to perform predictions for multiple
models), then you can use ``dc_model`` in combination with the ``with``
statement:

.. code:: python

    with dc_model:
        prediction = ds["ml_score_rbc"]  # same result as above

Please have a look at :ref:`this example <example_ml_builtin>` to see
dclab models in action.


The .modc file format
=====================

The .modc file format is not a reinvention of the wheel. It is merely
a wrapper around other ML file formats and describes which input
features (e.g. ``deform``, ``area_um``, ``image``, etc.) a machine learning
method maps onto which output features (e.g. ``ml_score_rbc``). A .modc file is
just a .zip file containing an index.json file that lists all
models. A model may be stored in multiple file formats (e.g. as a
`tensorflow SavedModel <https://www.tensorflow.org/guide/saved_model>`_
**and** as a Frozen Graph). Alongside the models, the .modc file format
also contains human-readable versions of the output features, SHA256
checksums, and the creation date:

.. code::

    example.modc (ZIP file contents)
    ├── index.json
    ├── model_0
    │         ├── another-format
    │         │        └── another_formats_file.suffix
    │         └── tensorflow-SavedModel.tf
    │             ├── assets
    │             ├── saved_model.pb
    │             └── variables
    │                 ├── variables.data-00000-of-00001
    │                 └── variables.index
    └── model_1
        └── tensorflow-SavedModel.tf
            ├── assets
            ├── saved_model.pb
            └── variables
                ├── variables.data-00000-of-00001
                └── variables.index

The corresponding index.json file could look like this:

.. code:: json

    {
      "model count": 2,
      "models": [
        {
          "date": "2020-11-03 17:01",
          "formats": {
            "tensorflow-SavedModel": "tensorflow-SavedModel.tf",
            "library-OtherFormat": "another-format"
          },
          "index": 0,
          "input features": [
            "deform"
          ],
          "name": "Example Model 1",
          "output features": [
            "ml_score_low",
            "ml_score_hig"
          ],
          "output labels": [
            "Low",
            "High"
          ],
          "path": "model_0",
          "sha256": "ec11c73ae870da4551d9fa9cc73271566b8f2356f284d4c2cb02057ecb5bf6ce"
        },
        {
          "date": "2020-11-03 17:02",
          "formats": {
            "tensorflow-SavedModel": "tensorflow-SavedModel.tf"
          },
          "index": 1,
          "input features": [
            "area_um",
            "image"
          ],
          "name": "Example Model 2",
          "output features": [
            "ml_score_rbc",
            "ml_score_sad"
          ],
          "output labels": [
            "red blood cells",
            "sad cells"
          ],
          "path": "model_1",
          "sha256": "ac43c73ae870da4551d9fa9cc73271566b8f2356f284d4c2cb02057ecb5ba812"
        }
      ]
    }

The great advantage of such a file format is that users can transparently
exchange machine learning methods and apply them in a reproducible manner to
any RT-DC dataset using dclab or Shape-Out.

To save a machine learning model to a .modc file, you can use the
`dclab.ml.save_modc` function:

.. code:: python

    dclab.ml.save_modc("path/to/file.modc", dc_model)

Conversely, you can load such a model at any time and use it for inference:

.. code:: python

    dc_model_loaded = dclab.ml.load_modc("path/to/file.modc")
    with dc_model_loaded:
        prediction = ds["ml_score_rbc"]  # same result as above


The methods for saving and loading .modc files are described in the
:ref:`code reference <sec_ref_ml_modc>`.


Helper functions
================

If you are working with `tensorflow <https://www.tensorflow.org/>`_,
you might find the functions in the :ref:`dclab.ml.tf_dataset
<sec_ref_ml_tf_dataset>` submodule helpful. Please also have a look
at the :ref:`machine-learning examples <example_ml_tensorflow>`.
