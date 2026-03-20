import warnings

from .kde.methods import (  # noqa: F401
    bin_num_doane, bin_width_doane, bin_width_percentile,
    ignore_nan_inf, kde_gauss, kde_histogram, kde_multivariate, kde_none,
    methods
)
from .kde.helpers import get_bad_vals  # noqa: F401


warnings.warn("`dclab.kde_methods` is deprecated; please use "
              "the `dclab.kde.methods` instead",
              DeprecationWarning)
