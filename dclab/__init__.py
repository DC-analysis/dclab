"""Core tools for the analysis of deformability cytometry datasets

Copyright (C) 2015 Paul Müller

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""
# flake8: noqa: F401
from . import definitions as dfn
from . import features
from . import isoelastics
from . import lme4
from .polygon_filter import PolygonFilter
from . import rtdc_dataset
from .rtdc_dataset import new_dataset, IntegrityChecker, RTDCWriter
from .rtdc_dataset.feat_temp import (
    register_temporary_feature, set_temporary_feature)
from .rtdc_dataset.feat_anc_ml import (
    MachineLearningFeature, load_modc, load_ml_feature, save_modc)
from .rtdc_dataset.feat_anc_plugin.plugin_feature import (
    PlugInFeature, load_plugin_feature)
from . import statistics
from . import util

from ._version import __version__, __version_tuple__


# Lazy-load deprecated kde modules 
kde_contours = util.LazyLoader("dclab.kde_contours")
kde_methods = util.LazyLoader("dclab.kde_methods")
