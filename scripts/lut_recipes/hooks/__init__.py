from . import le_2d_fem_19
from . import he_2d_fem_22
from . import he_3d_fem_22


lut_hooks = {
    "HE-2D-FEM-22": he_2d_fem_22,
    "HE-3D-FEM-22": he_3d_fem_22,
    "LE-2D-FEM-19": le_2d_fem_19,
}
