==============
Code reference
==============

.. toctree::
  :maxdepth: 2



.. _sec_ref_definitions:


Module-level methods
====================
.. autofunction:: dclab.new_dataset


Global definitions
==================
These definitionas are used throughout the dclab/Shape-In/Shape-Out ecosystem.

Metadata
--------
Valid configuration sections and keys are described in:
:ref:`sec_analysis_meta` and :ref:`sec_experiment_meta`.
You should use the following methods instead of accessing
the static metadata constants.

.. autofunction:: dclab.definitions.config_key_exists

.. autofunction:: dclab.definitions.get_config_value_descr

.. autofunction:: dclab.definitions.get_config_value_func

.. autofunction:: dclab.definitions.get_config_value_type


These constants are also available in the ``dclab.definitions`` module.

.. autodata:: dclab.definitions.meta_const.CFG_ANALYSIS
   :no-value:

.. autodata:: dclab.definitions.meta_const.CFG_METADATA
   :no-value:

.. autodata:: dclab.definitions.meta_const.config_keys
   :no-value:


Metadata parsers
----------------
.. automodule:: dclab.definitions.meta_parse
   :members:
   :undoc-members:


Features
--------
Features are discussed in more detail in :ref:`sec_features`.


.. autofunction:: dclab.definitions.check_feature_shape

.. autofunction:: dclab.definitions.feature_exists

.. autofunction:: dclab.definitions.get_feature_label

.. autofunction:: dclab.definitions.scalar_feature_exists


These constants are also available in the ``dclab.definitions`` module.

.. autodata:: dclab.definitions.feat_const.FEATURES_NON_SCALAR
   :no-value:

.. autodata:: dclab.definitions.feat_const.feature_names
   :no-value:

.. autodata:: dclab.definitions.feat_const.feature_labels
   :no-value:

.. autodata:: dclab.definitions.feat_const.feature_name2label
   :no-value:

.. autodata:: dclab.definitions.feat_const.scalar_feature_names
   :no-value:



.. _sec_ref_rtdc_dataset:

RT-DC dataset manipulation
==========================

Base class
----------

.. autoclass:: dclab.rtdc_dataset.RTDCBase
    :members:


HDF5 (.rtdc) format
-------------------

.. autoclass:: dclab.rtdc_dataset.RTDC_HDF5
    :members:

.. autoclass:: dclab.rtdc_dataset.fmt_hdf5.basin.HDF5Basin
    :members:

.. autodata:: dclab.rtdc_dataset.fmt_hdf5.MIN_DCLAB_EXPORT_VERSION



.. _sec_ref_rtdc_dataset_dcor:

DCOR (online) format
--------------------

.. autoclass:: dclab.rtdc_dataset.RTDC_DCOR
    :members:


.. autoclass:: dclab.rtdc_dataset.fmt_dcor.api.APIHandler
    :members:


HTTP (online) file format
-------------------------
.. automodule:: dclab.rtdc_dataset.fmt_http
    :members:


.. _sec_ref_rtdc_dataset_s3:

S3 (online) file format
-----------------------

.. automodule:: dclab.rtdc_dataset.fmt_s3
    :members:


Dictionary format
-----------------

.. autoclass:: dclab.rtdc_dataset.RTDC_Dict
    :members:

Hierarchy format
----------------

.. autoclass:: dclab.rtdc_dataset.RTDC_Hierarchy
    :members:


TDMS format
-----------

.. autoclass:: dclab.rtdc_dataset.RTDC_TDMS

.. autofunction:: dclab.rtdc_dataset.fmt_tdms.get_project_name_from_path

.. autofunction:: dclab.rtdc_dataset.fmt_tdms.get_tdms_files

.. _sec_ref_rtdc_config:


.. _cr_ancillaries:


Basin features
--------------
.. automodule:: dclab.rtdc_dataset.feat_basin
    :members:



Ancillaries
-----------
.. automodule:: dclab.rtdc_dataset.feat_anc_core.ancillary_feature
    :members:
    :undoc-members:


.. _cr_plugin_feat:

Plugin features
---------------
.. automodule:: dclab.rtdc_dataset.feat_anc_plugin.plugin_feature
    :members:
    :undoc-members:


.. _cr_temp_feat:

Temporary features
------------------
.. automodule:: dclab.rtdc_dataset.feat_temp
    :members:
    :undoc-members:


Config
------
.. autoclass:: dclab.rtdc_dataset.config.Configuration
    :members:

.. autofunction:: dclab.rtdc_dataset.config.load_from_file

.. _sec_ref_rtdc_export:

Export
------
.. autoexception:: dclab.rtdc_dataset.export.LimitingExportSizeWarning

.. autoclass:: dclab.rtdc_dataset.export.Export
    :members:


.. _sec_ref_rtdc_filter:

Filter
------
.. autoclass:: dclab.rtdc_dataset.filter.Filter
    :members:


Low-level functionalities
=========================

.. _sec_ref_downsampling:

downsampling
------------

.. automodule:: dclab.downsampling
   :members:
   :undoc-members:

   .. autofunction:: downsample_grid


.. _sec_ref_features:

features
--------

image-based
...........

.. autofunction:: dclab.features.contour.get_contour

.. autofunction:: dclab.features.bright.get_bright

.. autofunction:: dclab.features.inert_ratio.get_inert_ratio_cvx

.. autofunction:: dclab.features.inert_ratio.get_inert_ratio_raw

.. autofunction:: dclab.features.volume.get_volume

.. autofunction:: dclab.features.volume.counter_clockwise

.. autofunction:: dclab.features.volume.vol_revolve


emodulus
........

.. automodule:: dclab.features.emodulus
    :members:

.. automodule:: dclab.features.emodulus.load
    :members:

.. automodule:: dclab.features.emodulus.pxcorr
    :members:

.. automodule:: dclab.features.emodulus.scale_linear
    :members:

.. automodule:: dclab.features.emodulus.viscosity
    :members:


fluorescence
............

.. autofunction:: dclab.features.fl_crosstalk.correct_crosstalk

.. autofunction:: dclab.features.fl_crosstalk.get_compensation_matrix



.. _sec_ref_isoelastics:

isoelastics
-----------

.. automodule:: dclab.isoelastics
   :members:
   :undoc-members:



.. _sec_ref_kde_contours:

kde_contours
------------

.. automodule:: dclab.kde_contours
   :members:
   :undoc-members:


.. _sec_ref_kde_methods:

kde_methods
-----------

.. automodule:: dclab.kde_methods
   :members:
   :undoc-members:


.. _sec_ref_polygon_filter:

polygon_filter
--------------

.. automodule:: dclab.polygon_filter
   :members:
   :undoc-members:


.. _sec_ref_statistics:

statistics
----------

.. automodule:: dclab.statistics
   :members:
   :undoc-members:


.. _sec_ref_writer:


HDF5 manipulation
=================

.. automodule:: dclab.rtdc_dataset.copier
   :members:
   :undoc-members:


.. automodule:: dclab.rtdc_dataset.linker
   :members:
   :undoc-members:



Writing RT-DC files
===================

.. automodule:: dclab.rtdc_dataset.writer
   :members:
   :undoc-members:


.. _sec_ref_cli:

Command-line interface methods
==============================

.. automodule:: dclab.cli
   :members:
   :imported-members:


R and lme4
==========

.. _sec_ref_lme4:

.. automodule:: dclab.lme4.rlibs
   :members:
   :undoc-members:
   :exclude-members: MockPackage

.. automodule:: dclab.lme4.rsetup
   :members:
   :undoc-members:

.. automodule:: dclab.lme4.wrapr
   :members:
   :undoc-members:

