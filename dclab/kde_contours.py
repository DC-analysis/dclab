import warnings

from .kde.contours import (  # noqa: F401
    find_contours, find_contours_level, _find_quantile_level
)


warnings.warn("`dclab.kde_contours` is deptecated; please use "
              " the dclab.kde.contours instead",
              DeprecationWarning)
