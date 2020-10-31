.. _sec_av_ml:

================
Machine learning
================

To simplify machine-learning (ML) tasks in the context of RT-DC, dclab offers
a few helper functions. This section describes the recommended way
of implementing and distributing ML models based on RT-DC data. Please
make sure that you have installed dclab with the *ml* extra
(``pip install dclab[ml]``).


The .modc file format
=====================

The .modc file format is not a reinvention of the wheel. It is merely
a wrapper around other ML file formats that describes which input
features (e.g. ``deform``, ``area_um``, ``image``, etc.) a neural net
maps onto which output features (e.g. ``ml_score_rbc``). A .modc file is
a simple .zip file containing an index.json file that lists all
models. A model may be stored in multiple file formats (e.g. as a
`tensorflow SavedModel <https://www.tensorflow.org/guide/saved_model>`_
**and** as a Frozen Graph). Alongside the models, the .modc file format
also contains human-readable versions of the output features, SHA256
checksums, and the creation date.

The great advantage of such a file format is that users can transparently
exchange neural nets and apply them in a reproducible manner to any
RT-DC dataset using dclab or Shape-Out.

The methods for saving and loading .modc files are described in the
:ref:`code reference <sec_ref_ml_modc>`.


Helper functions
================

If you are working with `tensorflow <https://www.tensorflow.org/>`_,
you might find the functions in the :ref:`dclab.ml.tf_dataset
<sec_ref_ml_tf_dataset>` submodule helpful. Please also have a look
at the :ref:`machine-learning examples <example_ml_tensorflow>`.
