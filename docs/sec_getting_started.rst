===============
Getting started
===============

Installation
============

Dclab depends on several other Python packages:

 - `fcswrite <https://github.com/ZELLMECHANIK-DRESDEN/fcswrite>`_ (.fcs file export),
 - `h5py <http://www.h5py.org/>`_ (.rtdc file support).
 - `imageio <https://imageio.github.io/>`_ (.tdms file support, .avi file export),
 - `nptdms <http://nptdms.readthedocs.io/en/latest/>`_ (.tdms file support),
 - `numpy <https://docs.scipy.org/doc/numpy/>`_,
 - `scipy <https://docs.scipy.org/doc/scipy/reference/>`_,

In addition, dclab contains code from `OpenCV <https://opencv.org/>`_
(computation of moments) and `scikit-image <http://scikit-image.org/>`_
(computation of contours) to reduce the list of dependencies (these libraries
are not required by dclab).

To install dclab, use one of the following methods
(the above package dependencies will be installed automatically):
    
* from `PyPI <https://pypi.python.org/pypi/dclab>`_:
    ``pip install dclab``
* from `sources <https://github.com/ZellMechanik-Dresden/dclab>`_:
    ``pip install .`` or 
    ``python setup.py install``

Note that if you are installing from source or if no binary wheel is
available for your platform and Python version, `Cython <http://cython.org/>`_
will be installed to build the required dclab extensions. If this process
fails, please request a binary wheel for your platform (e.g. Windows 64bit)
and Python version (e.g. 3.6) by creating a new
`issue <https://github.com/ZellMechanik-Dresden/dclab/issues>`_.


Use cases
=========
If you are a frequent user of RT-DC, you might run into problems that
cannot (yet) be addressed with the graphical user interface
`Shape-Out <https://github.com/ZellMechanik-Dresden/ShapeOut>`_.
Here is a list of use cases that would motivate an installation of dclab.

- You would like to convert old .tdms-based datasets to the new .rtdc
  file format, because of enhanced speed in Shape-Out and reduced
  disk usage. What you are looking for is the command line program
  :ref:`sec_tdms2rtdc` that comes with dclab. It allows to batch-convert
  multiple measurements at a time. Note that you should keep the original
  .tdms files backed-up somewhere, because there might be future
  improvements or bug fixes from which you would like to benefit.
- You would like to apply a simple set of filters (e.g. polygon filters that you
  exported from within Shape-Out) to every new measurement you take and
  apply a custom data analysis pipeline to the filtered data. This is a
  straight-forward Python coding problem with dclab. After reading the
  basic usage section below, please have a look at the
  :ref:`polygon filter reference <sec_ref_polygon_filter>`.
- You would like to do advanced statistics or combine your RT-DC
  analysis with other fancy approaches such as machine-learning.
  It would be too laborious to do the analysis in Shape-Out, export the
  data as text files, and then open them in your custom Python script.
  If your initial analysis step with Shape-Out only involves tasks
  that can be automated, why not use dclab from the beginning? 
- You simulated RT-DC data and plan to import them in Shape-Out
  for testing. Once you have loaded your data as a numpy array, you
  can instantiate an :class:`RTDC_Dict <dclab.rtdc_dataset.RTDC_Dict>`
  class and then use the :class:`Export <dclab.rtdc_dataset.export.Export>`
  class to create an .rtdc data file.

If you are still unsure about whether to use dclab or not, you might
want to look at the :ref:`example section <sec_examples>`. If you need
advice, do not hesitate to
`create an issue <https://github.com/ZELLMECHANIK-DRESDEN/dclab/issues>`_.


Basic usage
===========
Experimental RT-DC datasets are always loaded with the
:func:`new_dataset <dclab.rtdc_dataset.load.new_dataset>` method:

.. code-block:: python

    import numpy as np
    import dclab

    # .tdms file format
    ds = dclab.new_dataset("/path/to/measurement/Online/M1.tdms")
    # .rtdc file format
    ds = dclab.new_dataset("/path/to/measurement/M2.rtdc")

The object returned by `new_dataset` is always an instance of
:class:`RTDCBase <dclab.rtdc_dataset.core.RTDCBase>`. To show all
available features, use:

.. code-block:: python

    print(ds.features)

This will list all scalar features (e.g. "area_um" and "deform") and all
non-scalar features (e.g. "contour" and "image"). Scalar features can be
filtered by editing the configuration of ``ds`` and calling ``ds.apply_filter()``:

.. code-block:: python

    # register filtering operations
    amin, amax = ds["area_um"].min(), ds["area_um"].max()
    ds.config["filtering"]["area_um min"] = (amax + amin) / 2
    ds.config["filtering"]["area_um max"] = amax
    ds.apply_filter()  # this step is important!

This will update the binary array ``ds.filter.all`` which can be used to
extract the filtered data:

.. code-block:: python

    area_um_filtered = ds["area_um"][ds.filter.all]

It is also possible to create a hierarchy child of this dataset that
only contains the filtered data.

.. code-block:: python

    ds_child = dclab.new_dataset(ds)

The hierarchy child ``ds_child`` is dynamic, i.e. when the filters in ``ds``
change, then ``ds_child`` also changes after calling ``ds_child.apply_filter()``.

Non-scalar features do not support fancy indexing (i.e.
``ds["image"][ds.filter.all]`` will not work. Use a for-loop to extract them.

.. code-block:: python

    for ii in range(len(ds)):
        image = ds["image"][ii]
        mask = ds["mask"][ii]
        # this is equivalent to ds["bright_avg"][ii]
        bright_avg = np.mean(image[mask])
        print("average brightness of event {}: {:.1f}".format(ii, bright_avg))

If you need more information to get started on your particular problem,
you might want to check out the :ref:`examples section <sec_examples>` and the
:ref:`advanced scripting section <sec_advanced_scripting>`.