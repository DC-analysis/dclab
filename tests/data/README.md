## Test data

This directory contains test data and other files for testing.

The data files are always stored in zip files (to reduce the size of the git
repository). The zip files are named according to file format (tdms or hdf5),
hint at possible use cases (e.g. fl for fluorescence), and contain a
representative data. Please follow that scheme when adding new test data.
To understand how loading test data works (basically, the archive is extracted
and the .tdms or .rtdc path returned), please take a look at the
`retrieve_data` function in the `helper_methods.py` file in the parent
directory .

Note that the tdms files (the container) are probably always broken, although
nptdms loads the data just fine. The reason for that is that Paul had no
means of writing data to tdms files back then and the original datasets were
just too big for a test case, so he just truncated the files. This is also
the reason why you have to `pytest.mark-ignore::` so many warnings for those
test data. In any case, the .tdms file format should be considered dead.
