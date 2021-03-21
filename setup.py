#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import exists, dirname, realpath
from setuptools import setup, Extension, find_packages
import sys

# numpy and cython are installed via pyproject.toml [build-system]
import numpy as np


maintainer = u"Paul Müller"
maintainer_email = "dev@craban.de"
description = 'Library for real-time deformability cytometry (RT-DC)'
name = "dclab"
year = "2015"

sys.path.insert(0, realpath(dirname(__file__))+"/"+name)
from _version import version  # noqa: E402


extras_require = {
        "dcor": ["requests"],
        "lme4": ["rpy2>=2.9.4"],
        "ml": ["tensorflow>=2.0"],
        "tdms": ["imageio[ffmpeg]",
                 "nptdms>=0.23.0",
                 ],
        "export": ["fcswrite>=0.5.0",  # fcs export
                   "imageio[ffmpeg]",  # avi export
                   ],
        }
# concatenate all other dependencies into "all"
extras_require["all"] = list(set(sum(list(extras_require.values()), [])))

setup(
    name=name,
    maintainer=maintainer,
    maintainer_email=maintainer_email,
    url='https://github.com/ZELLMECHANIK-DRESDEN/dclab',
    version=version,
    packages=find_packages(),
    package_dir={name: name},
    include_package_data=True,
    license="GPL v2",
    description=description,
    long_description=open('README.rst').read() if exists('README.rst') else '',
    install_requires=["h5py>=2.10.0",
                      "numpy>=1.17.0",  # numpy.random.default_rng
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
    python_requires=">=3.6",
    extras_require=extras_require,
    entry_points={
       "console_scripts": [
           "dclab-compress = dclab.cli:compress",
           "dclab-condense = dclab.cli:condense",
           "dclab-join = dclab.cli:join",
           "dclab-repack = dclab.cli:repack",
           "dclab-split = dclab.cli:split",
           "dclab-tdms2rtdc = dclab.cli:tdms2rtdc [tdms]",
           "dclab-verify-dataset = dclab.cli:verify_dataset",
            ],
       },
    keywords=["RT-DC", "deformation", "cytometry", "zellmechanik"],
    classifiers=['Operating System :: OS Independent',
                 'Programming Language :: Python :: 3',
                 'Topic :: Scientific/Engineering :: Visualization',
                 'Intended Audience :: Science/Research',
                 ],
    platforms=['ALL'],
    )
