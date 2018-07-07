==============
RT-DC datasets
==============
Knowing and understanding the :ref:`RT-DC dataset classes <sec_ref_rtdc_dataset>`
is the most important prerequisite when working with dclab. They are all
derived from :class:`RTDCBase <dclab.rtdc_dataset.core.RTDCBase>` which
gives access to feature with a dictionary interface, facilitates data export
and filtering, and comes with several convenience methods that are useful
for data visualization.
RT-DC datasets can be based on a data file format (:class:`RTDC_TDMS` and
:class:`RTDC_HDF5`), created from user-defined dictionaries (:class:`RTDC_Dict`),
or derived from other RT-DC datasets (:class:`RTDC_Hierarchy`).

