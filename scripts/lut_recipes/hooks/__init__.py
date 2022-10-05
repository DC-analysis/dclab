"""LUT hooks

This submodule contains manually-designed hooks for the different
LUTs used in dclab. Interpolating data on an irregular grid has its
pitfalls. Extracting isoelastics is tricky at the boundaries of the
simulation support. The main idea that is followed here is to
extrapolated isoelastics beyond the convec hull of the original
LUT to then get cleaner isoelastics up until the convex hull.
"""

from . import le_2d_fem_19
from . import he_2d_fem_22
from . import he_3d_fem_22


lut_hooks = {
    "HE-2D-FEM-22": he_2d_fem_22,
    "HE-3D-FEM-22": he_3d_fem_22,
    "LE-2D-FEM-19": le_2d_fem_19,
}
