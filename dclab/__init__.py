"""
This library contains classes and methods for the analysis
of real-time deformability cytometry (RT-DC) datasets.
"""
# flake8: noqa: F401
from . import definitions as dfn
from . import features
from . import isoelastics
from . import kde_contours
from . import kde_methods
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

from ._version import version as __version__
