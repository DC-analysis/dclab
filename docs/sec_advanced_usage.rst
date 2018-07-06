.. _sec_advanced_scripting:

==============
Advanced Usage
==============
This section motivates the design of dclab and highlights useful built-in
functionalities.


The dclab rationale
===================
When coding with dclab, you should be aware of the following definitions
and design principles.

Notation
--------
The following terms are used throughout this documentation and
the source code of dclab.

**Events**
    An event comprises all data recorded for the detection of one object
    (e.g. cell or bead) in an RT-DC measurement.

**Features**
    A feature is a measurement parameter of an RT-DC measurement. For
    instance, the feature "index" enumerates all recorded events, the
    feature "deformation" contains the deformation values of all events.
    There are scalar features, i.e. features that assign a single number
    to an event, and non-scalar features, such as "image" and "contour".
    The following features are supported by dclab:


    .. dclab_features:: scalar


    .. dclab_features:: non-scalar

**Ancillary features**
    Not all features available in dclab are recorded online during the
    acquisition of the experimental dataset. Some of the features are
    computed offline by dclab, such as "volume" or "emodulus". These
    ancillary features are computed on-the-fly and are made available
    seamlessly through the same feature interface.

**Filter**
    A filter can be used to gate events using features. There are
    min/max filters and 2D :ref:`polygon filters <sec_ref_polygon_filter>`.

**Metadata config**
    Every RT-DC measurement has metadata consisting of key-value-pairs.
    The following are supported:

    .. dclab_config:: metadata


**Analysis config**
    In addition to metadata, dclab also supports a user-defined analysis
    configuration which is usually part of a specific analysis pipeline
    and thus not considered to be metadata.

    .. dclab_config:: analysis


RT-DC datasets
--------------
Knowing and understanding the :ref:`RT-DC dataset classes <sec_ref_rtdc_dataset>`
is the most important prerequisite when working with dclab. They are all
derived from :class:`RTDCBase <dclab.rtdc_dataset.core.RTDCBase>` which
gives access to feature with a dictionary interface, facilitates data export
and filtering, and comes with several convenience methods that are useful
for data visualization.
RT-DC datasets can be based on a data file format (:class:`RTDC_TDMS` and
:class:`RTDC_HDF5`), created from user-defined dictionaries (:class:`RTDC_Dict`),
or derived from other RT-DC datasets (:class:`RTDC_Hierarchy`).


Data visualization
==================
For data visualization, dclab comes with a :ref:`sec_ref_downsampling` module
for reducing the number of plotted points while preserving regions with
few events and a :ref:`sec_ref_kde_methods` module for colorizing  events
according to event density. The functionalities of both modules are
made available directly via the :class:`RTDC_Base` class.

For data visualization, isoelasticity lines are often used to identify events
with similar elastic moduli. Isoelasticity lines are available via the
:ref:`sec_ref_isoelastics` module.


Statistics
==========
The :ref:`sec_ref_statistics` module comes with a predefined set of
methods to compute simple feature statistics. 

Data export
===========
The :class:`RTDC_Base` class has the attribute :class:`RTDC_Base.export`
which allows to export event data to several data file formats. See
:ref:`sec_ref_rtdc_export` for more information.

ShapeOut
========
Keep in mind that in some cases, it might still be useful to make use
of ShapeOut. For instance, you can create and export polygon filters
in ShapeOut and then import them in dclab.

