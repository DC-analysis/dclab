.. _sec_av_feat_plugin:

============================
User-defined plugin features
============================
For specialized applications, the features defined internally in dclab might
not be enough to describe certain aspects of your data. Plugin features
allow you to define a recipe for computing a new feature. This new feature
is then available *automatically* for *every* dataset loaded in dclab.

.. note::

    This will in future be supported by Shape-Out. If you would like to
    follow this development, you
    should subscribe to the issue about `plugin features in Shape-Out2
    <https://github.com/ZELLMECHANIK-DRESDEN/ShapeOut2/issues/85>`_.

.. note::

    The advantages of plugin features over :ref:`temporary features <sec_av_feat_temp>`
    are that plugin features are reproducible, shareable, versionable, and
    generally more transparent. You should only use temporary features if
    absolutely necessary.



Using plugin feature recipes
============================
If a colleague sent you a plugin feature recipe (a .py file), you just
have to load it in dclab to use it.

.. ipython::

    In [1]: import dclab

    In [2]: import numpy as np

    # load a plugin feature (makes `circ_times_area` available)
    In [3]: dclab.load_plugin_feature("data/example_plugin.py")

    # load some data
    In [4]: ds = dclab.new_dataset("data/example.rtdc")

    # access the new feature
    In [5]: circ_per_area = ds["circ_times_area"]

    # do some filtering
    In [7]: ds.config["filtering"]["circ_times_area min"] = 23

    In [8]: ds.config["filtering"]["circ_times_area max"] = 29

    In [9]: ds.apply_filter()

    In [10]: print("Removed {} out of {} events!".format(np.sum(~ds.filter.all), len(ds)))

Please also have a look at the :ref:`plugin usage example <example_plugin_usage>`.


Auto-loading multiple plugin feature recipes
============================================
If you have several plugins and would like to load them all at once,
you can do the following at the beginning of your scripts::

    for plugin_path in pathlib.Path("my_plugin_directory").rglob("*.py"):
        dclab.load_plugin_feature(plugin_path)


Writing a plugin feature recipe
===============================
A plugin feature recipe is defined in a Python script (e.g. `my_dclab_plugin.py`).
A plugin feature recipe contains a function and an ``info`` dictionary.
The function calculates the desired feature, and the dictionary defines
any extra (meta-)information of the feature. Both "method" (the function)
and "feature names" must be included in the ``info`` dictionary.
Note that many of the items in the dictionary must be lists!
Also note that a feature recipe may contain *multiple* features.
Below are three examples of creating and using plugin features.

.. note::

    Plugin features are based on :ref:`ancillary features <sec_features_ancillary>`
    (:ref:`code reference <cr_ancillaries>`).


Simple plugin feature recipe
----------------------------
In this :download:`basic example <data/example_plugin.py>`, the function
:func:`compute_my_feature` defines the basic feature `"circ_times_area"`.

.. literalinclude:: data/example_plugin.py
   :language: python


Advanced plugin feature recipe
------------------------------
In :download:`this example <../examples/plugin_example.py>`, the function
:func:`compute_some_new_features` defines two basic features:
`"circ_per_area"` and `"circ_times_area"`. Notice that both features are
computed in one function:

.. literalinclude:: ../examples/plugin_example.py
   :language: python

Here, all possible keys in the `info` dictionaryare shown (but not all are used).
The keys are additional keyword arguments to the
:class:`AncillaryFeature <dclab.rtdc_dataset.ancillaries.ancillary_feature.AncillaryFeature>`
class:

- ``features required`` corresponds to ``req_features``
- ``config required`` corresponds to ``req_config``
- ``method check required`` corresponds to ``req_func``

The ``scalar feature`` is a list of boolean values that defines whether
a feature is scalar or not (defaults to True).


.. _sec_av_feat_plugin_user_meta:

Plugin feature recipe with user-defined metadata
------------------------------------------------
In this :download:`example <data/example_plugin_metadata.py>`, the function
:func:`compute_area_exponent` defines the basic feature `area_exp`,
which is calculated using
:ref:`user-defined metadata<sec_user_meta>`.

.. literalinclude:: data/example_plugin_metadata.py
   :language: python

The above plugin uses the "exp" key in the "user" configuration section
to set the exponent value (notice the ``"config required"`` key in the ``info`` dict).
Therefore, the feature `area_exp` is only available, when
``rtdc_ds.config["user"]["exp"]`` is set.

    .. ipython::

        In [1]: import dclab

        In [2]: dclab.load_plugin_feature("data/example_plugin_metadata.py")

        In [3]: ds = dclab.new_dataset("data/example.rtdc")

        # The plugin feature is not yet available, because "user:exp" is missing
        In [4]: "area_exp" in ds
        Out[4]: False

        # Set user-defined metadata
        In [5]: my_metadata = {"inlet": True, "n_channels": 4, "exp": 3}

        In [6]: ds.config["user"] = my_metadata

        # The plugin feature is now available
        In [7]: "area_exp" in ds
        Out[7]: True

        # Now the plugin feature can be accessed like any regular feature
        In [8]: area_exp = ds["area_exp"]


.. _sec_av_feat_plugin_reload:

Reloading plugin features stored in data files
==============================================
It is also possible to store plugin features within datasets on disk.
This may be useful if the speed of calculation of your plugin feature is
slow, and you don't want to recalculate each time you open your dataset.
The process for storing plugin feature data is similar to that :ref:`described
for temporary features <sec_av_feat_temp_store>`.
If you would like to access those feature data at a later time point,
you still have to load the plugin feature recipe first::

    dclab.load_plugin_feature("/path/to/plugin.py")
    ds = dclab.new_dataset("/path/to/data_with_new_plugin_feature.rtdc")
    circ_per_area = ds["circ_per_area"]

And this works as well (loading plugin after instantiation)::

    ds = dclab.new_dataset("/path/to/data_with_new_plugin_feature.rtdc")
    dclab.load_plugin_feature("/path/to/plugin.py")
    circ_per_area = ds["circ_per_area"]

.. note::

    After storing and reloading, this feature is now an `innate` feature.
    You could in principle also access it by registering it as a temporary
    feature (e.g. if you don't have the recipe lying around).

See the :ref:`code reference on plugin features <cr_plugin_feat>` for more
information.
