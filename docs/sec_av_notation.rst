.. _sec_av_notation:

========
Notation
========
When coding with dclab, you should be aware of the following definitions
and design principles.

Events
------
An event comprises all data recorded for the detection of one object
(e.g. cell or bead) in an RT-DC measurement.


.. _sec_features:

Features
--------
A feature is a measurement parameter of an RT-DC measurement. For
instance, the feature "index" enumerates all recorded events, the
feature "deform" contains the deformation values of all events.
There are scalar features, i.e. features that assign a single number
to an event, and non-scalar features, such as "image" and "contour".
The following features are supported by dclab:

.. _sec_features_scalar:

Scalar features
...............

.. dclab_features:: scalar

In addition to these scalar features, it is possible to define
a large number of features dedicated to machine-learning, the
"ml_score\_???" features: The "?" can be a digit or a lower-case
letter of the alphabet, e.g. "ml\_score\_rbc" or "ml\_score_3a3".
If "ml_score\_???" features are defined, then the ancillary
"ml_class" feature, which identifies the most-probable feature
for each event, becomes available.  

.. _sec_features_non_scalar:

Non-scalar features
...................

.. dclab_features:: non-scalar


Examples
........

**deformation vs. area plot** 

    .. plot::
        
        import matplotlib.pylab as plt
        import dclab
        ds = dclab.new_dataset("data/example.rtdc")
        ax = plt.subplot(111)
        ax.plot(ds["area_um"], ds["deform"], "o", alpha=.2)
        ax.set_xlabel(dclab.dfn.get_feature_label("area_um"))
        ax.set_ylabel(dclab.dfn.get_feature_label("deform"))
        plt.show()

**event image plot**

    .. plot::
        
        import matplotlib.pylab as plt
        import dclab
        ds = dclab.new_dataset("data/example_video.rtdc")
        ax1 = plt.subplot(211, title="image")
        ax2 = plt.subplot(212, title="mask")
        ax1.imshow(ds["image"][6], cmap="gray")
        ax2.imshow(ds["mask"][6])

.. _sec_features_ancillary:

Ancillary features
------------------
Not all features available in dclab are recorded online during the
acquisition of the experimental dataset. Some of the features are
computed offline by dclab, such as "volume", "emodulus", or
scores from imported machine learning models ("ml_score_xxx"). These
ancillary features are computed on-the-fly and are made available
seamlessly through the same interface.


Filters
-------
A filter can be used to gate events using features. There are
min/max filters and 2D :ref:`polygon filters <sec_ref_polygon_filter>`.
The following table defines the main filtering parameters:

.. dclab_config:: filtering

Min/max filters are also defined in the *filters* section:

.. csv-table::
    :header: filtering, explanation
    :widths: 30, 70

    area_um min,  Exclude events with area [µm²] below this value
    area_um max, Exclude events with area [µm²] above this value
    aspect max, Exclude events with an aspect ratio above this value
    ..., ...

Examples
........

**excluding events with large deformation**

    .. plot::
        
        import matplotlib.pylab as plt
        import dclab
        ds = dclab.new_dataset("data/example.rtdc")

        ds.config["filtering"]["deform min"] = 0
        ds.config["filtering"]["deform max"] = .1
        ds.apply_filter()
        dif = ds.filter.all

        f, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        axes[0].plot(ds["area_um"], ds["bright_avg"], "o", alpha=.2)
        axes[0].set_title("unfiltered")
        axes[1].plot(ds["area_um"][dif], ds["bright_avg"][dif], "o", alpha=.2)
        axes[1].set_title("Deformation <= 0.1")

        for ax in axes:
            ax.set_xlabel(dclab.dfn.get_feature_label("area_um"))
            ax.set_ylabel(dclab.dfn.get_feature_label("bright_avg"))

        plt.tight_layout()
        plt.show()


**excluding random events**

    This is useful if you need to have a (sub-)dataset of a specified
    size. The downsampling is reproducible (the same points are excluded).

    .. plot::
        
        import matplotlib.pylab as plt
        import dclab
        ds = dclab.new_dataset("data/example.rtdc")
        ds.config["filtering"]["limit events"] = 4000
        ds.apply_filter()
        fid = ds.filter.all
        
        ax = plt.subplot(111)
        ax.plot(ds["area_um"][fid], ds["deform"][fid], "o", alpha=.2)
        ax.set_xlabel(dclab.dfn.get_feature_label("area_um"))
        ax.set_ylabel(dclab.dfn.get_feature_label("deform"))
        plt.show()

.. _sec_experiment_meta:

Experiment metadata
-------------------
Every RT-DC measurement has metadata consisting of key-value-pairs.
The following are supported:

.. dclab_config:: metadata

**Example**: date and time of a measurement

    .. ipython::
    
        In [1]: import dclab

        In [2]: ds = dclab.new_dataset("data/example.rtdc")

        In [3]: ds.config["experiment"]["date"], ds.config["experiment"]["time"]

.. _sec_analysis_meta:

Analysis metadata
-----------------
In addition to inherent (defined during data acquisition) metadata,
dclab also supports additional metadata that are relevant for certain
data analysis pipelines, such as Young's modulus computation or
fluorescence crosstalk correction.

.. dclab_config:: calculation

.. _sec_user_meta:

User-defined metadata
---------------------
In addition to the registered metadata keys listed above,
you may also define custom metadata in the "user" section.
This section will be saved alongside the other metadata when
a dataset is exported as an .rtdc (HDF5) file.

.. note::
    It is recommended to use the following data types for the value of
    each key: ``str``, ``bool``, ``float`` and ``int``. Other data types may
    not render nicely in ShapeOut2 or DCOR.

To edit the "user" section in dclab, simply modify the `config`
property of a loaded dataset. The changes made are *not* written
to the underlying file.

**Example**: Setting custom "user" metadata in dclab

    .. ipython::

        In [1]: import dclab

        In [2]: ds = dclab.new_dataset("data/example.rtdc")

        In [3]: my_metadata = {"inlet": True, "n_channels": 4}

        In [4]: ds.config["user"] = my_metadata

        In [5]: other_metadata = {"outlet": False, "RBC": True}

        # we can also add metadata with the `update` method
        In [6]: ds.config["user"].update(other_metadata)

        # or
        In [7]: ds.config.update({"user": other_metadata})

        In [8]: print(ds.config["user"])

        # we can clear the "user" section like so:
        In [9]: ds.config["user"].clear()

If you are implementing a custom data acquisition pipeline, you may
alternatively add user-defined meta data (permanently) to an .rtdc file
in a post-measurement step like so.

**Example**: Setting custom "user" metadata permanently

   .. code::

       import h5py
       with h5py.File("/path/to/your/dataset.rtdc") as h5:
           h5.attrs["user:inlet"] = True
           h5.attrs["user:n_channels"] = 4
           h5.attrs["user:outlet"] = False
           h5.attrs["user:RBC"] = True
           h5.attrs["user:project"] = "strangelove"

User-defined metadata can also be used with user-defined
:ref:`plugin features <sec_av_feat_plugin_user_meta>`. This allows you
to design plugin features which utilize your pipeline-specific metadata.
