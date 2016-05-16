#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import exists, dirname, realpath
from setuptools import setup
import sys

author = u"Paul Müller"
authors = [author]
description = 'Data analysis for real-time deformability cytometry.'
name = 'dclab'
year = "2015"


sys.path.insert(0, realpath(dirname(__file__))+"/"+name)
try:
    from _version import version  # @UnresolvedImport
except:
    version = "unknown"


if __name__ == "__main__":
    setup(
        name=name,
        author=author,
        author_email='paul.mueller@biotec.tu-dresden.de',
        url='https://github.com/ZellMechanik-Dresden/dclab',
        version=version,
        packages=[name],
        package_dir={name: name},
        license="GPL v2",
        description=description,
        long_description=open('README.rst').read() if exists('README.rst') else '',
        install_requires=[
                          "nptdms",
                          "NumPy >= 1.5.1",
                          "scipy",
                          "statsmodels >= 0.5.0"
                          ],
        keywords=["RTDC", "deformation", "cytometry", "zellmechanik"],
        setup_requires=['pytest-runner'],
        tests_require=["pytest", "urllib3"],
        package_data={'dclab': ['dclab/dclab.cfg']},
        include_package_data=True,
        classifiers= [
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.2',
            'Programming Language :: Python :: 3.3',
            'Topic :: Scientific/Engineering :: Visualization',
            'Intended Audience :: Science/Research'
                     ],
        platforms=['ALL'],
        )

