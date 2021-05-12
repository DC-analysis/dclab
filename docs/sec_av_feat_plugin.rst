.. _sec_av_feat_plugin:

============================
User-defined plugin features
============================
For specialized applications, the features defined internally in dclab might
not be enough to describe certain aspects of your data. You can define a plugin
feature in your dataset without modifying any files on disk. Another option is
to define a :ref:`temporary feature <sec_av_feat_temp>`.


Writing a plugin feature script
===============================
To create a dclab plugin feature, you define a function and a dictionary.
The function will calculate the desired feature, while the dictionary will
simply gather all the extra information useful when creating a feature.

In this example, the function :func:`compute_some_new_features` defines a
basic "circ_per_area" feature. Then, the :dict:`info` dictionary is filled in
by the user. Both "method" and "feature names" must be included in the
:dict:`info` dictionary. Note that many of the items in the dictionary must be
lists!

.. code-block:: python

    def compute_some_new_features(rtdc_ds):
        """The function that does the heavy-lifting"""
        circ_per_area = rtdc_ds["circ"] / rtdc_ds["area_um"]
        # returns a dictionary-like object
        return {"circ_per_area": circ_per_area}


    info = {
        "method": compute_some_new_features,
        "description": "This plugin will compute some features",
        "long description": "Even longer description that "
                            "can span multiple lines",
        "feature names": ["circ_per_area"],
        "feature labels": ["Circularity per Area"],
        "features required": ["circ", "area_um"],
        "config required": [],
        "method check required": lambda x: True,
        "scalar feature": [True],
        "version": "0.1.0",
    }

The above code can be downloaded
:ref:`here <sec_examples.html#Plugin Feature>`_. Once downloaded, place the
file in a suitable folder on your computer, e.g.,
`/Documents/dclab_plugins/plugin_example_features.py`.


Setting a plugin feature in a dataset
========================================
For this example, you can register the temporary feature `fl1_mean` and
manually set a corresponding filter for your dataset.

.. ipython::

    In [1]: import dclab

    # load a single plugin feature
    In [2]: dclab.load_plugin_feature("/path/to/plugin_example_features.py")

    # load some data
    In [3]: ds = dclab.new_dataset("path/to/rtdc/file")

    # access the first feature
    In [4]: circ_per_area = ds["circ_per_area"]

    # access the other feature
    In [5]: circ_times_area = ds["circ_times_area"]

    # do some filtering
    In [6]: ds.config["filtering"]["circ_per_area min"] = 4

    In [7]: ds.config["filtering"]["circ_per_area max"] = 200

    In [8]: ds.apply_filter()

    In [9]: print("Removed {} out of {} events!".format(np.sum(~ds.filter.all), len(ds)))


Accessing temporary features stored in data files
=================================================
It is also possible to store temporary features within datasets on disk.
At a later time point, you can then load this data file from disk with access
to those temporary features [1]_.

.. note::

    This is definitely not supported by Shape-Out, DCKit, or DCOR/DCOR-Aid.
    If you would like to compute additional features with Shape-Out, you
    should subscribe to the `issue about PluginFeature
    <https://github.com/ZELLMECHANIK-DRESDEN/dclab/issues/105>`_.

There are two ways of adding temporary features to an .rtdc data file.

- 1. With `h5py <https://docs.h5py.org>`_:

    .. code:: python

        import dclab
        import h5py
        import numpy as np

        # extract the feature data from the dataset
        with dclab.new_dataset("/path/to/data.rtdc") as ds:
            fl1_mean = np.array([np.mean(ds["trace"]["fl1_raw"][ii]) for ii in range(len(ds))])

        # write the feature to the HDF5 file
        with h5py.File("/path/to/data.rtdc", "a") as h5:
            h5["events"]["fl1_mean"] = fl1_mean

- 2. Via :func:`RTDCBase.export.hdf5 <dclab.rtdc_dataset.export.Export.hdf5>`:

    .. code:: python

        import dclab
        import h5py
        import numpy as np

        # register temporary feature
        dclab.register_temporary_feature(feature="fl1_mean")

        with dclab.new_dataset("/path/to/data.rtdc") as ds:
            # extract the feature information from the dataset
            fl1_mean = np.array([np.mean(ds["trace"]["fl1_raw"][ii]) for ii in range(len(ds))])
            # set the data
            dclab.set_temporary_feature(rtdc_ds=ds, feature="fl1_mean", data=fl1_mean)
            # export the data to a new file
            ds.export.hdf5("/path/to/data_with_fl1_mean.rtdc",
                           features=ds.features_innate + ["fl1_mean"])

If you wish to load the data at a later time point, you have to make sure
that you register the temporary feature before trying to access it.
This will not work::

    ds = dclab.new_dataset("/path/to/data_with_fl1_mean.rtdc")
    fl1_mean = ds["fl1_mean"]

But this works::

    dclab.register_temporary_feature(feature="fl1_mean")
    ds = dclab.new_dataset("/path/to/data_with_fl1_mean.rtdc")
    fl1_mean = ds["fl1_mean"]

And this works as well (registering after instantiation)::

    ds = dclab.new_dataset("/path/to/data_with_fl1_mean.rtdc")
    dclab.register_temporary_feature(feature="fl1_mean")
    fl1_mean = ds["fl1_mean"]


Please read the :ref:`code reference on temporary features
<cr_temp_feat>` for more
information.

.. [1] I know, storing *temporary* features on disk sounds like a
       counter-intuitive concept, but this is a very convenient extension
       of temporary features which came with almost no overhead.
       In a sense, it's still temporary, because you always have to register
       the feature before you can access it.



See the examples section for more complicated examples
In this example, the function :func:`compute_some_new_features` defines
two features at once

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

