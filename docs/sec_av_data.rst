===============
Data processing
===============

Visualization
=============
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


Export
======
The :class:`RTDC_Base` class has the attribute :class:`RTDC_Base.export`
which allows to export event data to several data file formats. See
:ref:`sec_ref_rtdc_export` for more information.


ShapeOut
========
Keep in mind that in some cases, it might still be useful to make use
of ShapeOut. For instance, you can create and export polygon filters
in ShapeOut and then import them in dclab.



