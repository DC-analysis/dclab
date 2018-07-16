==============
RT-DC datasets
==============
Knowing and understanding the :ref:`RT-DC dataset classes <sec_ref_rtdc_dataset>`
is an important prerequisite when working with dclab. They are all
derived from :class:`RTDCBase <dclab.rtdc_dataset.RTDCBase>` which
gives access to feature with a dictionary-like interface, facilitates data export
and filtering, and comes with several convenience methods that are useful
for data visualization.
RT-DC datasets can be based on a data file format
(:class:`RTDC_TDMS <dclab.rtdc_dataset.RTDC_TDMS>` and
:class:`RTDC_HDF5 <dclab.rtdc_dataset.RTDC_HDF5>`), created from user-defined
dictionaries (:class:`RTDC_Dict <dclab.rtdc_dataset.RTDC_Dict>`),
or derived from other RT-DC datasets
(:class:`RTDC_Hierarchy <dclab.rtdc_dataset.RTDC_Hierarchy>`).


Loading data from disk
======================
The convenience function :func:`dclab.new_dataset` takes care of determining
the data file format (tdms or hdf5) and returns the corresponding derived
class.

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


Creating hierarchies
--------------------
When applying filtering operations, it is sometimes helpful to
use hierarchies for keeping track of the individual filtering steps.

.. ipython::

    In [7]: child = dclab.new_dataset(ds)

    In [8]: grandchild = dclab.new_dataset(child)

    In [9]: ds.config["filtering"]["deform max"] = .15

    In [10]: child.config["filtering"]["area_um max"] = 80

    In [11]: grandchild.apply_filter()

    In [12]: len(ds), len(child), len(grandchild)

    In [13]: ds.filter.all.sum(), child.filter.all.sum(), grandchild.filter.all.sum()


Note that calling ``ds1_b.apply_filter()`` automatically calls
``ds1_a.apply_filter()`` and ``ds1.apply_filter()``. Also note that,
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

    # available features
    In [16]: ds.features

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
