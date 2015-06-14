#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import join, dirname, realpath
from setuptools import setup, Command
import subprocess as sp
import sys

author = u"Paul Müller"
authors = [author]
description = 'Data analysis for real-time deformability cytometry.'
name = 'dclab'
year = "2015"


sys.path.insert(0, realpath(dirname(__file__))+"/"+name)
try:
    from _version import version
except:
    version = "unknown"


class PyTest(Command):
    """ Perform pytests
    """
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        errno = sp.call([sys.executable, 'tests/runtests.py'])
        raise SystemExit(errno)


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
        long_description=open(join(dirname(__file__), 'README.md')).read(),
        install_requires=[
                          "nptdms",
                          "NumPy >= 1.5.1",
                          "scipy",
                          "statsmodels"
                          ],
        keywords=["RTDC", "deformation", "cytometry", "zellmechanik"],
        extras_require={
                        'doc': ['sphinx']
                       },
        #data_files=[('dclab', ['dclab/dclab.cfg'])],
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
        cmdclass = {'test': PyTest,
                    },
        )

