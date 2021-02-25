.. _sec_av_datasets:

==============
RT-DC datasets
==============
Knowing and understanding the :ref:`RT-DC dataset classes <sec_ref_rtdc_dataset>`
is an important prerequisite when working with dclab. They are all
derived from :class:`RTDCBase <dclab.rtdc_dataset.RTDCBase>` which
gives access to features with a dictionary-like interface, facilitates data export
or filtering, and comes with several convenience methods that are useful
for data visualization.
RT-DC datasets can be based on a data file format
(:class:`RTDC_TDMS <dclab.rtdc_dataset.RTDC_TDMS>` and
:class:`RTDC_HDF5 <dclab.rtdc_dataset.RTDC_HDF5>`), accessed
from an online repository (:class:`RTDC_HDF5 <dclab.rtdc_dataset.RTDC_DCOR>`),
created from user-defined
dictionaries (:class:`RTDC_Dict <dclab.rtdc_dataset.RTDC_Dict>`),
or derived from other RT-DC datasets
(:class:`RTDC_Hierarchy <dclab.rtdc_dataset.RTDC_Hierarchy>`).


Basic usage
===========
The convenience function :func:`dclab.new_dataset` takes care of determining
the data format and returns the corresponding derived class.

.. ipython::

    In [1]: import dclab

    In [2]: ds = dclab.new_dataset("data/example.rtdc")

    In [3]: ds.__class__.__name__


Working with other data
-----------------------
It is also possible to load other data into dclab from a dictionary.

.. ipython::

    In [4]: data = dict(deform=np.random.rand(100),
       ...:             area_um=np.random.rand(100))
       ...:

    In [5]: ds_dict = dclab.new_dataset(data)

    In [6]: ds_dict.__class__.__name__


Using filters
-------------
Filters are used to mask e.g. debris or doublets from a dataset.

.. ipython::

    # Restrict the deformation to 0.15
    In [6]: ds.config["filtering"]["deform min"] = 0

    In [7]: ds.config["filtering"]["deform max"] = .15

    # Manually excluding events using array indices is also possible:
    # `ds.filter.manual` is a 1D boolean array of size `len(ds)`
    # where `False` values mean that the events are excluded.
    In [8]: ds.filter.manual[[0, 400, 345, 1000]] = False

    In [9]: ds.apply_filter()

    # The boolean array `ds.filter.all` represents the applied filter
    # and can be used for indexing.
    In [9]: ds["deform"].mean(), ds["deform"][ds.filter.all].mean()

Note that ``ds.apply_filter()`` must be called, otherwise
``ds.filter.all`` will not be updated.

Creating hierarchies
--------------------
When applying filtering operations, it is sometimes helpful to
use hierarchies for keeping track of the individual filtering steps.

.. ipython::

    In [5]: child = dclab.new_dataset(ds)

    In [6]: child.config["filtering"]["area_um min"] = 0

    In [7]: child.config["filtering"]["area_um max"] = 80

    In [8]: grandchild = dclab.new_dataset(child)

    In [11]: grandchild.apply_filter()

    In [12]: len(ds), len(child), len(grandchild)

    In [13]: ds.filter.all.sum(), child.filter.all.sum(), grandchild.filter.all.sum()


Note that calling ``grandchild.apply_filter()`` automatically calls
``child.apply_filter()`` and ``ds.apply_filter()``. Also note that,
as expected, the size of each hierarchy child is identical to the sum of the
boolean filtering array from its hierarchy parent.


Scripting goodies
-----------------
Here are a few useful functionalities for scripting with dclab.

.. ipython::

    # unique identifier of the RTDCBase instance (not reproducible)
    In [14]: ds.identifier

    # reproducible hash of the dataset
    In [15]: ds.hash

    # dataset format
    In [15]: ds.format

    # all available features
    In [16]: ds.features

    # scalar (one number per event) features
    In [16]: ds.features_scalar

    # innate (present in the underlying data file) features
    In [16]: ds.features_innate

    # loaded (innate and computed ancillaries) features
    In [16]: ds.features_loaded

    # test feature availability (success)
    In [17]: "area_um" in ds

    # test feature availability (failure)
    In [18]: "image" in ds

    # accessing a feature and computing its mean
    In [19]: ds["area_um"].mean()

    # accessing the measurement configuration
    In [20]: ds.config.keys()

    In [21]: ds.config["experiment"]

    # determine the identifier of the hierarchy parent
    In [22]: child.config["filtering"]["hierarchy parent"]

    

Statistics
==========
The :ref:`sec_ref_statistics` module comes with a predefined set of
methods to compute simple feature statistics. 


.. ipython::

    In [1]: import dclab

    In [2]: ds = dclab.new_dataset("data/example.rtdc")

    In [3]: stats = dclab.statistics.get_statistics(ds,
       ...:                                         features=["deform", "aspect"],
       ...:                                         methods=["Mode", "Mean", "SD"])
       ...:

    In [4]: dict(zip(*stats))


Note that the statistics take into account the applied filters:

.. ipython::

    In [4]: ds.config["filtering"]["deform min"] = 0

    In [5]: ds.config["filtering"]["deform max"] = .1

    In [6]: ds.apply_filter()

    In [7]: stats2 = dclab.statistics.get_statistics(ds,
       ...:                                          features=["deform", "aspect"],
       ...:                                          methods=["Mode", "Mean", "SD"])
       ...:

    In [8]: dict(zip(*stats2))


These are the available statistics methods:

.. ipython::

    In [9]: dclab.statistics.Statistics.available_methods.keys()


Export
======
The :class:`RTDCBase <dclab.rtdc_dataset.RTDCBase>` class has the attribute
:attr:`RTDCBase.export <dclab.rtdc_dataset.RTDCBase.export>`
which allows to export event data to several data file formats. See
:ref:`sec_ref_rtdc_export` for more information.

.. ipython::

    In [9]: ds.export.tsv(path="export_example.tsv",
       ...:               features=["area_um", "deform"],
       ...:               filtered=True,
       ...:               override=True)
       ...:

    In [9]: ds.export.hdf5(path="export_example.rtdc",
       ...:                features=["area_um", "aspect", "deform"],
       ...:                filtered=True,
       ...:                override=True)
       ...:

Note that data exported as HDF5 files can be loaded with dclab
(reproducing the previously computed statistics - without filters).

.. ipython::

    In [11]: ds2 = dclab.new_dataset("export_example.rtdc")

    In [12]: ds2["deform"].mean()

