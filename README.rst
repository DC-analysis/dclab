dclab
=====

|PyPI Version| |Build Status Unix| |Build Status Win| |Coverage Status| |Docs Status|


This is a Python library for the post-measurement analysis of
real-time deformability cytometry (RT-DC) datasets; an essential part of
`Shape-Out <https://github.com/ZELLMECHANIK-DRESDEN/ShapeOut>`__.

Documentation
-------------

The documentation, including the code reference and examples, is available at
`dclab.readthedocs.io <https://dclab.readthedocs.io/en/stable/>`__.


Installation
------------
To install the latest release, simply run

::

	pip install dclab


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

**Very important:** Please always try to use 

::

	git pull --rebase

instead of

::

	git pull
	
to prevent confusions in the commit history.

Tests
~~~~~
dclab is tested using pytest. If you have the time, please write test
methods for your code and put them in the ``tests`` directory.


Incrementing version
~~~~~~~~~~~~~~~~~~~~
Dclab currently gets its version from the latest git tag.
If you think that a new version should be published,
create a tag on the master branch (if you have the necessary
permissions to do so):

::

	git tag -a "0.1.3"
	git push --tags origin

Appveyor and TravisCI will then automatically build source package and wheels 
and publish them on PyPI.


Notes on `Shape-Out <https://github.com/ZELLMECHANIK-DRESDEN/ShapeOut/>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The `appveyor Shape-Out build <https://ci.appveyor.com/project/paulmueller/ShapeOut>`__
is automatically triggered after each commit to the Shape-Out repository. At each build,
the master branch of dclab is checked out and the Shape-Out installer is built with it.
Therefore, it is not necessary to bump the version of dclab or to upload the latest
version of dclab to PyPI in order to get your new code into Shape-Out.


.. |PyPI Version| image:: https://img.shields.io/pypi/v/dclab.svg
   :target: https://pypi.python.org/pypi/dclab
.. |Build Status Unix| image:: https://img.shields.io/travis/ZELLMECHANIK-DRESDEN/dclab.svg?label=build_linux_osx
   :target: https://travis-ci.org/ZELLMECHANIK-DRESDEN/dclab
.. |Build Status Win| image:: https://img.shields.io/appveyor/ci/paulmueller/dclab/master.svg?label=build_win
   :target: https://ci.appveyor.com/project/paulmueller/dclab
.. |Coverage Status| image:: https://img.shields.io/codecov/c/github/ZELLMECHANIK-DRESDEN/dclab/master.svg
   :target: https://codecov.io/gh/ZELLMECHANIK-DRESDEN/dclab
.. |Docs Status| image:: https://readthedocs.org/projects/dclab/badge/?version=latest
   :target: https://readthedocs.org/projects/dclab/builds/
