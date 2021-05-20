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

Configuration
-------------
Valid configuration sections and keys are described in:
:ref:`sec_analysis_meta` and :ref:`sec_experiment_meta`.


.. autodata:: dclab.definitions.CFG_ANALYSIS
   :no-value:

.. autodata:: dclab.definitions.CFG_METADATA
   :no-value:

.. autodata:: dclab.definitions.config_funcs
   :no-value:

.. autodata:: dclab.definitions.config_keys
   :no-value:

.. autodata:: dclab.definitions.config_types
   :no-value:


Features
--------
Features are discussed in more detail in: :ref:`sec_features`.


.. autofunction:: dclab.definitions.feature_exists

.. autofunction:: dclab.definitions.get_feature_label

.. autofunction:: dclab.definitions.scalar_feature_exists


.. autodata:: dclab.definitions.FEATURES_NON_SCALAR
   :no-value:

.. autodata:: dclab.definitions.feature_names
   :no-value:

.. autodata:: dclab.definitions.feature_labels
   :no-value:

.. autodata:: dclab.definitions.feature_name2label
   :no-value:

.. autodata:: dclab.definitions.scalar_feature_names
   :no-value:

Parse functions
---------------
.. automodule:: dclab.parse_funcs
   :members:
   :undoc-members:


.. _sec_ref_rtdc_dataset:

RT-DC dataset manipulation
==========================

Base class
----------

.. autoclass:: dclab.rtdc_dataset.RTDCBase
    :members:


.. _sec_ref_rtdc_dataset_dcor:

DCOR (online) format
--------------------

.. autoclass:: dclab.rtdc_dataset.RTDC_DCOR
    :members:


.. autoclass:: dclab.rtdc_dataset.fmt_dcor.APIHandler
    :members:



Dictionary format
-----------------

.. autoclass:: dclab.rtdc_dataset.RTDC_Dict

HDF5 (.rtdc) format
-------------------

.. autoclass:: dclab.rtdc_dataset.RTDC_HDF5
    :members:

.. autodata:: dclab.rtdc_dataset.fmt_hdf5.MIN_DCLAB_EXPORT_VERSION


Hierarchy format
----------------

.. autoclass:: dclab.rtdc_dataset.RTDC_Hierarchy

TDMS format
-----------

.. autoclass:: dclab.rtdc_dataset.RTDC_TDMS

.. autofunction:: dclab.rtdc_dataset.fmt_tdms.get_project_name_from_path

.. autofunction:: dclab.rtdc_dataset.fmt_tdms.get_tdms_files

.. _sec_ref_rtdc_config:


.. _cr_ancillaries:

Ancillaries
-----------
.. automodule:: dclab.rtdc_dataset.ancillaries.ancillary_feature
    :members:
    :undoc-members:


.. _cr_plugin_feat:

Plugin features
---------------
.. automodule:: dclab.rtdc_dataset.plugins.plugin_feature
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



Machine learning
================

.. _sec_ref_ml_mllibs:

.. automodule:: dclab.ml.mllibs
   :members:
   :undoc-members:
   :exclude-members: MockPackage

.. _sec_ref_ml_modc:

.. automodule:: dclab.ml.modc
   :members:
   :undoc-members:

.. _sec_ref_ml_models:

.. automodule:: dclab.ml.models
   :members:
   :undoc-members:

.. _sec_ref_ml_tf_dataset:

.. automodule:: dclab.ml.tf_dataset
   :members:
   :undoc-members:
