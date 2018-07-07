========
Notation
========
When coding with dclab, you should be aware of the following definitions
and design principles.

Events
------
An event comprises all data recorded for the detection of one object
(e.g. cell or bead) in an RT-DC measurement.


Features
--------
A feature is a measurement parameter of an RT-DC measurement. For
instance, the feature "index" enumerates all recorded events, the
feature "deformation" contains the deformation values of all events.
There are scalar features, i.e. features that assign a single number
to an event, and non-scalar features, such as "image" and "contour".
The following features are supported by dclab:

.. dclab_features:: scalar

.. dclab_features:: non-scalar


Ancillary features
------------------
Not all features available in dclab are recorded online during the
acquisition of the experimental dataset. Some of the features are
computed offline by dclab, such as "volume" or "emodulus". These
ancillary features are computed on-the-fly and are made available
seamlessly through the same feature interface.


Filters
-------
A filter can be used to gate events using features. There are
min/max filters and 2D :ref:`polygon filters <sec_ref_polygon_filter>`.

.. csv-table::
    :header: feature filter, explanation
    :widths: 30, 70

    area_um min,  Exclude events with area [µm²] below this value
    area_um max, Exclude events with area [µm²] above this value
    aspect max, Exclude events with an aspect ratio above this value
    ..., ...


Experiment metadata
-------------------
Every RT-DC measurement has metadata consisting of key-value-pairs.
The following are supported:

.. dclab_config:: metadata


Analysis metadata
-----------------
In addition to metadata, dclab also supports a user-defined analysis
configuration which is usually part of a specific analysis pipeline
and thus not considered to be metadata.

.. dclab_config:: analysis
