|dclab|
=======

|PyPI Version| |Build Status Unix| |Build Status Win| |Coverage Status| |Docs Status|


This is a Python library for the post-measurement analysis of
real-time deformability cytometry (RT-DC) datasets; an essential part of
`Shape-Out <https://github.com/ZELLMECHANIK-DRESDEN/ShapeOut2>`__.

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
`Changelog <https://github.com/ZELLMECHANIK-DRESDEN/dclab/blob/master/CHANGELOG>`__. 

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
    flake8 dclab
    flake8 docs
    flake8 examples
    flake8 tests


Incrementing version
~~~~~~~~~~~~~~~~~~~~
Dclab gets its version from the latest git tag.
If you think that a new version should be published,
create a tag on the master branch (if you have the necessary
permissions to do so)::

    git tag -a "0.1.3"
    git push --tags origin

Appveyor and TravisCI will then automatically build source package and wheels 
and publish them on PyPI.


.. |dclab| image:: https://raw.github.com/ZELLMECHANIK-DRESDEN/dclab/master/docs/logo/dclab.png
.. |PyPI Version| image:: https://img.shields.io/pypi/v/dclab.svg
   :target: https://pypi.python.org/pypi/dclab
.. |Build Status Unix| image:: https://img.shields.io/github/workflow/status/ZELLMECHANIK-DRESDEN/dclab/Checks
   :target: https://github.com/ZELLMECHANIK-DRESDEN/dclab/actions?query=workflow%3AChecks
.. |Build Status Win| image:: https://img.shields.io/appveyor/ci/paulmueller/dclab/master.svg?label=build_win
   :target: https://ci.appveyor.com/project/paulmueller/dclab
.. |Coverage Status| image:: https://img.shields.io/codecov/c/github/ZELLMECHANIK-DRESDEN/dclab/master.svg
   :target: https://codecov.io/gh/ZELLMECHANIK-DRESDEN/dclab
.. |Docs Status| image:: https://readthedocs.org/projects/dclab/badge/?version=latest
   :target: https://readthedocs.org/projects/dclab/builds/
