from setuptools import Extension, setup

import numpy as np


setup(
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
        ]
)
