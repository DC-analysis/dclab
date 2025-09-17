import warnings

from .kde.contours import (  # noqa: F401
    find_contours_level, _find_quantile_level, get_quantile_levels
)


warnings.warn("`dclab.kde_contours` is deprecated; please use "
              "the `dclab.kde.contours` instead",
              DeprecationWarning)
