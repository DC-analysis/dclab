|dclab|
=======

|PyPI Version| |Build Status| |Coverage Status| |Docs Status|


This is a Python library for the post-measurement analysis of
real-time deformability cytometry (RT-DC) datasets; an essential part of
the DC Cosmos (
`DCscope <https://github.com/DC-analysis/DCscope>`__,
`DCOR <https://github.com/DCOR-dev/dcor_control>`__,
`DCOR-Aid <https://github.com/DCOR-dev/DCOR-Aid>`__,
`DCTag <https://github.com/DC-analysis/DCTag>`__,
`DCKit <https://github.com/DC-analysis/DCKit>`__,
).

Documentation
-------------
The documentation, including the code reference and examples, is available at
`dclab.readthedocs.io <https://dclab.readthedocs.io/en/stable/>`__.


Installation
------------

::

    pip install dclab[all]

For more options, please check out the `documentation
<https://dclab.readthedocs.io/en/latest/sec_getting_started.html#installation>`__.


Information for developers
--------------------------


Contributing
~~~~~~~~~~~~
The main branch for developing dclab is master.
If you want to make small changes like one-liners,
documentation, or default values in the configuration,
you may work on the master branch. If you want to change
more, please (fork dclab and) create a separate branch,
e.g. ``my_new_feature_dev``, and create a pull-request
once you are done making your changes.
Please make sure to edit the 
`Changelog <https://github.com/DC-analysis/dclab/blob/master/CHANGELOG>`__.

**Very important:** Please always try to use ::


    git pull --rebase

instead of::

    git pull

to prevent non-linearities in the commit history.

Tests
~~~~~
dclab is tested using pytest. If you have the time, please write test
methods for your code and put them in the ``tests`` directory. To run
the tests, install `pytest` and run::

    pytest tests


Docs
~~~~
The docs are built with `sphinx <https://www.sphinx-doc.org>`_. Please make
sure they compile when you change them (this also includes function doc strings)::

    cd docs
    pip install -r requirements.txt
    sphinx-build . _build  # open "index.html" in the "_build" directory


PEP8
~~~~
We use flake8 to enforce coding style::

    pip install flake8
    flake8 --exclude _version.py dclab
    flake8 docs
    flake8 examples
    flake8 tests


Type Hinting
~~~~~~~~~~~~
For type hinting and docstring styling, try to keep the 
Code Reference readable in the documentation website.
If in doubt just ask, or look at the examples in the codebase.

* Simple type combintions such as ``float`` or ``str | pathlib.Path``
  should be included as type hints, but do not need to be included
  in the docstring parameter description.
* More involved type hints can have extra information in the 
  docstring. For example for numpy arrays, ``npt.NDArray[np.bool]`` doesn't
  render in a readable way in the Code Reference, and doesn't include shape.
  Therefore, you can also keep the docstring parameter description e.g.,
  ``binary ndarray of shape (M, N)``.


Incrementing version
~~~~~~~~~~~~~~~~~~~~
Dclab gets its version from the latest git tag.
If you think that a new version should be published,
create a tag on the master branch (if you have the necessary
permissions to do so)::

    git tag -a "0.1.3"
    git push --tags origin

Appveyor and GitHub Actions will then automatically build source package and wheels
and publish them on PyPI.


.. |dclab| image:: https://raw.github.com/DC-analysis/dclab/master/docs/logo/dclab.png
.. |PyPI Version| image:: https://img.shields.io/pypi/v/dclab.svg
   :target: https://pypi.python.org/pypi/dclab
.. |Build Status| image:: https://img.shields.io/github/actions/workflow/status/DC-analysis/dclab/check.yml
   :target: https://github.com/DC-analysis/dclab/actions?query=workflow%3AChecks
.. |Coverage Status| image:: https://img.shields.io/codecov/c/github/DC-analysis/dclab/master.svg
   :target: https://codecov.io/gh/DC-analysis/dclab
.. |Docs Status| image:: https://readthedocs.org/projects/dclab/badge/?version=latest
   :target: https://readthedocs.org/projects/dclab/builds/
