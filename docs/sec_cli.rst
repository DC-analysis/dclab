======================
Command-line interface
======================


.. note::

   You may also call all of these command-line functions from within
   Python. For instance, to compress a dataset, you would use
   :func:`dclab.cli.compress`:

   .. code-block:: python

       import dclab.cli
       dclab.cli.compress(
           path_out="/path/to/compressed_file.rtdc",
           path_in="/path/to/original.rtdc")

   For more information please take a look at the code reference
   of the :ref:`CLI submodule <sec_ref_cli>`.


.. _sec_compress:

dclab-compress
--------------

.. simple_argparse::
   :module: dclab.cli
   :func: compress_parser
   :prog: dclab-compress


.. _sec_condense:

dclab-condense
--------------

.. simple_argparse::
   :module: dclab.cli
   :func: condense_parser
   :prog: dclab-condense


.. _sec_join:

dclab-join
----------

.. simple_argparse::
   :module: dclab.cli
   :func: join_parser
   :prog: dclab-join


.. _sec_repack:

dclab-repack
------------

.. simple_argparse::
   :module: dclab.cli
   :func: repack_parser
   :prog: dclab-repack


.. _sec_split:

dclab-split
-----------

.. simple_argparse::
   :module: dclab.cli
   :func: split_parser
   :prog: dclab-split


.. _sec_tdms2rtdc:

dclab-tdms2rtdc
---------------

.. simple_argparse::
   :module: dclab.cli
   :func: tdms2rtdc_parser
   :prog: dclab-tdms2rtdc
    

.. _sec_verify_dataset:

dclab-verify-dataset
--------------------

.. simple_argparse::
   :module: dclab.cli
   :func: verify_dataset_parser
   :prog: dclab-verify-dataset
