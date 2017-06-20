#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This library contains classes and methods for the analysis
of real-time deformability cytometry (RT-DC) data sets.
"""
from __future__ import division, print_function, unicode_literals

import os

# Definitions
from . import definitions as dfn
from ._version import version as __version__
from .rtdc_dataset import new_dataset
from .polygon_filter import PolygonFilter
from . import statistics, elastic
