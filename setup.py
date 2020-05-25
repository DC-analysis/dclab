#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import exists, dirname, realpath
from setuptools import setup, Extension, find_packages
import sys

# numpy and cython are installed via pyproject.toml [build-system]
import numpy as np

author = u"Paul Müller"
authors = [author]
description = 'Library for real-time deformability cytometry (RT-DC)'
name = 'dclab'
year = "2015"


sys.path.insert(0, realpath(dirname(__file__))+"/"+name)
try:
    from _version import version  # @UnresolvedImport
except BaseException:
    version = "unknown"


extras_require = {
        "dcor": ["requests"],
        "tdms": ["nptdms>=0.23.0",  # read tdms files
                 "imageio>=2.3.0,<2.5.0;python_version<'3.4'",  # avi (old)
                 "imageio[ffmpeg]>=2.8.0;python_version>='3.4'",  # avi (new)
                 ],
        "export": ["fcswrite>=0.5.0",  # fcs export
                   "imageio>=2.3.0,<2.5.0;python_version<'3.4'",  # avi (old)
                   "imageio[ffmpeg]>=2.8.0;python_version>='3.4'",  # avi (new)
                   ],
        "all": ["fcswrite>=0.5.0",
                "imageio>=2.3.0,<2.5.0;python_version<'3.4'",  # avi (old)
                "imageio[ffmpeg]>=2.8.0;python_version>='3.4'",  # avi (new)
                "nptdms>=0.23.0",
                "requests",
                ]
        }


setup(
    name=name,
    author=author,
    author_email='dev@craban.de',
    url='https://github.com/ZELLMECHANIK-DRESDEN/dclab',
    version=version,
    packages=find_packages(),
    package_dir={name: name},
    include_package_data=True,
    license="GPL v2",
    description=description,
    long_description=open('README.rst').read() if exists('README.rst') else '',
    install_requires=["h5py>=2.10.0",
                      "numpy>=1.16.0",  # cython includes
                      "pathlib;python_version<='3.4'",
                      "backports.functools_lru_cache;python_version<'3.3'",
                      "scipy>=0.14.0",
                      ],
    ext_modules=[
        Extension("dclab.external.skimage._shared.geometry",
                  sources=["dclab/external/skimage/_shared/geometry.pyx"],
                  include_dirs=[np.get_include()]
                  ),
        Extension("dclab.external.skimage._find_contours_cy",
                  sources=["dclab/external/skimage/_find_contours_cy.pyx"],
                  include_dirs=[np.get_include()]
                  ),
        Extension("dclab.external.skimage._pnpoly",
                  sources=["dclab/external/skimage/_pnpoly.pyx"],
                  include_dirs=[np.get_include(), "_shared"]
                  ),
        ],
    # not to be confused with definitions in pyproject.toml [build-system]
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "urllib3"] + extras_require["all"],
    extras_require=extras_require,
    entry_points={
       "console_scripts": [
           "dclab-compress = dclab.cli:compress",
           "dclab-condense = dclab.cli:condense",
           "dclab-join = dclab.cli:join",
           "dclab-repack = dclab.cli:repack",
           "dclab-tdms2rtdc = dclab.cli:tdms2rtdc [tdms]",
           "dclab-verify-dataset = dclab.cli:verify_dataset",
            ],
       },
    keywords=["RT-DC", "deformation", "cytometry", "zellmechanik"],
    classifiers=['Operating System :: OS Independent',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Scientific/Engineering :: Visualization',
                 'Intended Audience :: Science/Research',
                 ],
    platforms=['ALL'],
    )
