# flake8: noqa: F401
import warnings

from .base import KernelDensityEstimator

warnings.warn(
    "Modules have been moved to 'kde'. Update your imports accordingly.",
    DeprecationWarning,
    stacklevel=2
)