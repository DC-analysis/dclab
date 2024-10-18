"""A wrapper around R with the lme4 package"""
from . import rsetup, wrapr  # noqa: F401
from .wrapr import Rlme4, bootstrapped_median_distributions  # noqa: F401
from .rsetup import (  # noqa: F401
    set_r_lib_path, get_r_path, get_r_version, require_lme4, set_r_path)
