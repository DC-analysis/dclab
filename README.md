## dclab
[![PyPI](http://img.shields.io/pypi/v/dclab.svg)](https://pypi.python.org/pypi/dclab)

This is a Python library for the post-measurement
analysis of real-time deformability cytometry (RT-DC)
data sets; an essential part of
[ShapeOut](https://github.com/ZellMechanik-Dresden/ShapeOut).


### Installation
To install the latest release, simply run `pip install dclab`.
If you want to get the bleeding edge from GitHub, run
`pip install git+git://github.com/ZellMechanik-Dresden/dclab.git`.


### Information for developers


#### Contributing
The main branch for developing dclab is master.
If you want to make small changes like one-liners,
documentation, or default values in the configuration,
you may work on the master branch. If you want to change
more, please (fork dclab and) create a separate branch,
e.g. `my_new_feature_dev`, and create a pull-request
once you are done making your changes.
**Please make sure to edit the 
[Changelog](https://github.com/ZellMechanik-Dresden/dclab/blob/master/CHANGELOG)**. 
**Very important:** Please always try use 

	git pull --rebase

instead of

	git pull
	
to prevent confusions in the commit history.

#### Tests
dclab is tested using pytest. If you have the time, please write test
methods for your code and put them in the `tests` directory.


#### Incrementing version
dclab currently gets its version from 
[dclab._version](https://github.com/ZellMechanik-Dresden/dclab/blob/master/dclab/_version.py).
If you think that a new version should be published,
increment the version string in that file and create
a tag on the master branch (if you have the necessary
permissions to do so):

	git tag -a "0.1.3"
	git push --tags origin


#### Uploading to PyPI
If this is not automated yet, only @paulmueller can upload
a new version of dclab to the Python Package Index.


#### Notes on [ShapeOut](https://github.com/ZellMechanik-Dresden/ShapeOut/)
The [appveyor ShapeOut build](https://ci.appveyor.com/project/paulmueller/ShapeOut)
is automatically triggered after each commit to the ShapeOut repository. At each build,
the master branch of dclab is checked out and the ShapeOut installer is built with it.
Therefore, it is not necessary to bump the version of dclab or to upload the latest
version of dclab to PyPI in order to get your new code into ShapeOut.





