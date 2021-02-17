"""A wrapper around R with the lme4 package"""
# flake8: noqa: F401
from . import rlibs, rsetup, wrapr
from .wrapr import Rlme4, bootstrapped_median_distributions
from .rsetup import get_r_path, get_r_version, install_lme4, set_r_path
