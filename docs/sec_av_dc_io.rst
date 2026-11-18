.. _sec_av_dc_io:

===========
DC data I/O
===========

When working with DC data, you will inevitably run into the situation where
you would like to write some part of dataset (be it an .rtdc file or data
on DCOR) to a new file. Depending on the situation, one or more of the
following subsection will probably cover what you need.


.. _sec_av_dc_io_export:

Exporting data
==============
The :class:`RTDCBase <.RTDCBase>` class has the attribute
:attr:`RTDCBase.export <.RTDCBase.export>` which allows to export event
data to several data file formats. See :ref:`sec_ref_rtdc_export` for more
information.

.. ipython::

    In [1]: import dclab

    In [2]: ds = dclab.new_dataset("data/example.rtdc")

    # Restrict the deformation to 0.15
    In [3]: ds.config["filtering"]["deform min"] = 0

    In [4]: ds.config["filtering"]["deform max"] = .15

    In [5]: ds.apply_filter()

    In [6]: print("Deformation mean before filtering:", ds["deform"][:].mean())

    In [7]: print("Deformation mean after filtering:", ds["deform"][ds.filter.all].mean())

    # Export to .tsv
    In [8]: ds.export.tsv(path="export_example.tsv",
       ...:               features=["area_um", "deform"],
       ...:               filtered=True,
       ...:               override=True)
       ...:

    # Export to .rtdc
    In [9]: ds.export.hdf5(path="export_example.rtdc",
       ...:                features=["area_um", "aspect", "deform"],
       ...:                filtered=True,
       ...:                override=True)
       ...:

Note that data exported as HDF5 files can be loaded with dclab
(reproducing the previously computed statistics - without filters).

.. ipython::

    In [10]: ds2 = dclab.new_dataset("export_example.rtdc")

    In [11]: ds2["deform"][:].mean()


.. _sec_av_dc_io_write:

Writing to an .rtdc file
========================

If you would like to create your own .rtdc files, you can
make use of the :class:`RTDCWriter <dclab.rtdc_dataset.writer.RTDCWriter>` class.

.. ipython::

    In [4]: with dclab.RTDCWriter("my-data.rtdc", mode="reset") as hw:
       ...:     hw.store_metadata({"experiment": {"sample": "my sample",
       ...:                                       "run index": 1}})
       ...:     hw.store_feature("deform", np.random.rand(100))
       ...:     hw.store_feature("area_um", np.random.rand(100))

    In [5]: ds_custom = dclab.new_dataset("my-data.rtdc")

    In [6]: print(ds_custom.features)

    In [7]: print(ds_custom.config["experiment"])


The `mode` argument defines how data should be written to the file. In the
above example, any existing data in *my-data-rtdc* is deleted. If you
are writing to an existing file, you may use `mode="append"` to append
events to existing features or `mode="replace"` to replace entire features
with new data. Use `"append"` if you are continuously writing new events
to a file. Use `"replace"` if you are rewriting feature data (e.g. in
a script that computes features for all events).

.. warning::

    If you are using the wrong mode, you can introduce inconsistencies.
    Say you are computing custom feature data and you changed the algorithm
    that does the computation. You have an existing .rtdc file with the
    outdated `userdef1` feature. You would like to replace this data with
    the data from the updated algorithm. If you open that file with
    `mode="append"` and write `userdef1` to it, then the `userdef1` feature
    will have twice the length. You should used `mode="replace"` instead.


.. _sec_av_dc_io_copy:

Copying (parts of) a dataset
============================

In some situations, you would only like to copy an entire feature column
from one dataset to a new file without modification. The :mod:`copier
<dclab.rtdc_dataset.copier>` submodule enables this on a low-level.

- Use the :func:`.rtdc_copy` method to create a compressed version of a DC
  dataset opened as an HDF5 file (:class:`.RTDC_HDF5` or :class:`.RTDC_S3`).
- Use the :func:`.h5ds_copy` method to copy parts of an HDF5 dataset to
  another HDF5 file, with the option to enforce compression (if the source
  :class:`h5py.Dataset` is not compressed properly already).


.. _sec_av_dc_io_vid2dc:

Converting video files to .rtdc
===============================

If you have a video file of cells flowing through a channel (e.g. from a
custom setup or a legacy .tdms measurement), you can convert it to a raw
.rtdc file with the script
`vid2dc.py <https://github.com/DC-analysis/dclab/blob/main/scripts/vid2dc.py>`_
that is part of the dclab repository. The resulting .rtdc file only
contains the "image" feature and basic metadata. You can then use
`ChipStream <https://github.com/DC-analysis/ChipStream>`_ to perform
background correction, segmentation, and feature extraction.

.. warning::

    Most video formats use lossy compression, which alters the image data
    (e.g. compression artifacts at the cell boundaries). This affects the
    features computed from these images. If possible, record your data
    directly in a lossless format (e.g. .rtdc) and only use this script
    for data that are not available otherwise.

The script requires a few additional packages::

    pip install dclab[tdms] imageio[pyav] click

To convert a single video file, run::

    python vid2dc.py path/to/video.avi

This creates a file ``path/to/video_raw.rtdc``. You may also specify
the output path as a second argument. If you pass a directory, all video
files in it (.avi, .mp4, .mpg, .wmv, .mkv, including subdirectories)
are converted (without an output path). Files that were converted before
are not converted again, unless the input file or its metadata changed.
If a file cannot be converted, the error is written to a
``*_vid2dc_error.log`` file next to it.

The metadata are taken from files next to the video file, if available
(e.g. the "M1_camera.ini" and "M1_para.ini" files of a .tdms measurement,
or an .ini file with the same name as the video file). Otherwise, the
script uses default values, e.g. a frame rate of 2000 fps, a pixel size of
0.34 µm, and a flow rate of 0. Please check the metadata of the
resulting .rtdc file and correct them if necessary (e.g. with
:class:`RTDCWriter <dclab.rtdc_dataset.writer.RTDCWriter>` in
``mode="append"``), because features such as the area in µm² or the
Young's modulus depend on them.
