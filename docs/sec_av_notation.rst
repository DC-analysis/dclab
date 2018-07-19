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

.. dclab_features:: scalar

.. _sec_features_non_scalar:

.. dclab_features:: non-scalar

**Example**: deformation vs. area plot 

    .. plot::
        
        import matplotlib.pylab as plt
        import dclab
        ds = dclab.new_dataset("data/example.rtdc")
        ax = plt.subplot(111)
        ax.plot(ds["area_um"], ds["deform"], "o", alpha=.2)
        ax.set_xlabel(dclab.dfn.feature_name2label["area_um"])
        ax.set_ylabel(dclab.dfn.feature_name2label["deform"])
        plt.show()

**Example**: event image plot

    .. plot::
        
        import matplotlib.pylab as plt
        import dclab
        ds = dclab.new_dataset("data/example_video.rtdc")
        ax1 = plt.subplot(211, title="image")
        ax2 = plt.subplot(212, title="mask")
        ax1.imshow(ds["image"][6], cmap="gray")
        ax2.imshow(ds["mask"][6])


Ancillary features
------------------
Not all features available in dclab are recorded online during the
acquisition of the experimental dataset. Some of the features are
computed offline by dclab, such as "volume" or "emodulus". These
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

**Example**: excluding events with large deformation 

    .. plot::
        
        import matplotlib.pylab as plt
        import dclab
        ds = dclab.new_dataset("data/example.rtdc")

        ds.config["filtering"]["deform max"] = .1
        ds.apply_filter()
        dif = ds.filter.all

        f, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        axes[0].plot(ds["area_um"], ds["bright_avg"], "o", alpha=.2)
        axes[0].set_title("unfiltered")
        axes[1].plot(ds["area_um"][dif], ds["bright_avg"][dif], "o", alpha=.2)
        axes[1].set_title("Deformation <= 0.1")

        for ax in axes:
            ax.set_xlabel(dclab.dfn.feature_name2label["area_um"])
            ax.set_ylabel(dclab.dfn.feature_name2label["bright_avg"])

        plt.tight_layout()
        plt.show()


**Example**: excluding random events
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
        ax.set_xlabel(dclab.dfn.feature_name2label["area_um"])
        ax.set_ylabel(dclab.dfn.feature_name2label["deform"])
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
In addition to metadata, dclab also supports a user-defined analysis
configuration which is usually part of a specific analysis pipeline
and thus not considered to be experimental metadata.

.. dclab_config:: calculation
