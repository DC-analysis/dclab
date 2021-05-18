.. _sec_av_feat_plugin:

============================
User-defined plugin features
============================
For specialized applications, the features defined internally in dclab might
not be enough to describe certain aspects of your data. One solution to this problem
are :ref:`temporary features <sec_av_feat_temp>`. However, temporary features
require you to manually set the feature data for each dataset and thus are only
applicable to very specialized scenarios. Plugin features allow you to define a
recipe for computing a feature, which is then applied to *every* dataset
*automatically*. Plugin features are based on
:ref:`ancillary features <sec_features_ancillary>`.


Writing a plugin feature script
===============================
A plugin feature is defined in a Python script (e.g. `my_dclab_plugin.py`).
A plugin feature script contains a function and a dictionary.
The function will calculate the desired feature, and the dictionary defines
the extra information useful when creating a feature.
Both "method" and "feature names" must be included in the dictionary.
Note that many of the items in the dictionary must be lists!
Below are two examples of creating and using plugin features.


Simple plugin feature example
-----------------------------
In this basic example, the function :func:`compute_a_new_feature` defines the
basic feature `"circ_per_area"`.

.. code-block:: python

    def compute_a_new_feature(rtdc_ds):
        """The function that does the heavy-lifting"""
        circ_per_area = rtdc_ds["circ"] / rtdc_ds["area_um"]
        # returns a dictionary-like object
        return {"circ_per_area": circ_per_area}


    info = {
        "method": compute_some_new_features,
        "description": "This plugin computes the area per circularity",
        "feature names": ["circ_per_area"],
        "features required": ["circ", "area_um"],
        "version": "0.1.0",
    }


Advanced plugin feature example
-------------------------------
In this example, the function :func:`compute_some_new_features` defines two
basic features: `"circ_per_area"` and `"circ_times_area"`. Notice that both
features are computed in one function. In this example, all of the
:dict:`info` dictionary metadata keys are shown.

.. code-block:: python

    def compute_some_new_features(rtdc_ds):
        """The function that does the heavy-lifting"""
        circ_per_area = rtdc_ds["circ"] / rtdc_ds["area_um"]
        circ_times_area = rtdc_ds["circ"] * rtdc_ds["area_um"]
        # returns a dictionary-like object
        return {"circ_per_area": circ_per_area, "circ_times_area": circ_times_area}


    info = {
        "method": compute_some_new_features,
        "description": "This plugin will compute some features",
        "long description": "Even longer description that "
                            "can span multiple lines",
        "feature names": ["circ_per_area", "circ_times_area"],
        "feature labels": ["Circularity per Area", "Circularity times Area"],
        "features required": ["circ", "area_um"],
        "config required": [],
        "method check required": lambda x: True,
        "scalar feature": [True, True],
        "version": "0.1.0",
    }


This example plugin can be downloaded
:ref:`here <sec_examples.html#Plugin Feature>`_ in the file
"plugin_example_features.py".


Setting a plugin feature in a dataset
=====================================
For this example, you can register the plugin features `"circ_per_area"` and
`"circ_times_area"` that are defined in the plugin script. Then, set a
corresponding filter for your dataset.

.. ipython::

    In [1]: import dclab

    In [2]: import numpy as np

    # load a single plugin feature
    In [3]: dclab.load_plugin_feature("/path/to/plugin.py")

    # load some data
    In [4]: ds = dclab.new_dataset("/path/to/rtdc/file")

    # access the first feature
    In [5]: circ_per_area = ds["circ_per_area"]

    # access the other feature
    In [6]: circ_times_area = ds["circ_times_area"]

    # do some filtering
    In [7]: ds.config["filtering"]["circ_times_area min"] = 23

    In [8]: ds.config["filtering"]["circ_times_area max"] = 29

    In [9]: ds.apply_filter()

    In [10]: print("Removed {} out of {} events!".format(np.sum(~ds.filter.all), len(ds)))


Accessing plugin features stored in data files
==============================================
It is also possible to store plugin features within datasets on disk.
At a later time point, you can then load this data file from disk with access
to those plugin features.

.. note::

    This will in future be supported by Shape-Out. If you would like to
    follow this development, you
    should subscribe to the `issue about PluginFeature
    <https://github.com/ZELLMECHANIK-DRESDEN/dclab/issues/105>`_.

There are two ways of adding plugin features to an .rtdc data file.

- 1. With `h5py <https://docs.h5py.org>`_:

    .. code:: python

        import dclab
        import h5py

        # load plugin features from script
        dclab.load_plugin_feature("/path/to/plugin.py")

        # extract the feature data from the dataset
        with dclab.new_dataset("/path/to/data.rtdc") as ds:
            circ_per_area = ds["circ_per_area"]
            circ_times_area = ds["circ_times_area"]

        # write the feature to the HDF5 file
        with h5py.File("/path/to/data.rtdc", "a") as h5:
            h5["events"]["circ_per_area"] = circ_per_area
            h5["events"]["circ_times_area"] = circ_times_area

- 2. Via :func:`RTDCBase.export.hdf5 <dclab.rtdc_dataset.export.Export.hdf5>`:

    .. code:: python

        import dclab
        import h5py

        # load plugin features from script
        dclab.load_plugin_feature("/path/to/plugin.py")

        with dclab.new_dataset("/path/to/data.rtdc") as ds:
            # export the data to a new file
            ds.export.hdf5("/path/to/data_with_new_plugin_feature.rtdc",
                           features=ds.features_innate + ["circ_per_area",
                                                          "circ_times_area"])


If you wish to load the data at a later time point, the plugin needs
to be loaded again before accessing its data.::

    dclab.load_plugin_feature("/path/to/plugin.py")
    ds = dclab.new_dataset("/path/to/data_with_new_plugin_feature.rtdc")
    circ_per_area = ds["circ_per_area"]

And this works as well (loading plugin after instantiation)::

    ds = dclab.new_dataset("/path/to/data_with_new_plugin_feature.rtdc")
    dclab.load_plugin_feature("/path/to/plugin.py")
    circ_per_area = ds["circ_per_area"]


See the :ref:`code reference on plugin features <cr_plugin_feat>` for more
information.


Loading multiple plugin features
================================

If you have several plugins and would like to load them all at once,
you can do the following::

    for plugin_path in pathlib.Path("my_plugin_directory").rglob("*.py"):
        dclab.load_plugin_feature(plugin_path)

