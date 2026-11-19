import warnings

from .kde.contours import (  # noqa: F401
    find_contours_level, get_quantile_levels
)


warnings.warn("`dclab.kde_contours` is deprecated; please use "
              "`dclab.kde.contours` instead",
              DeprecationWarning)
