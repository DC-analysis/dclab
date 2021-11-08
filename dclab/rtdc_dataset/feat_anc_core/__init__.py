from .ancillary_feature import AncillaryFeature  # noqa: F401
from . import af_basic
from . import af_emodulus
from . import af_fl_max_ctc
from . import af_image_contour
from . import af_ml_class
from . import af_ml_score  # noqa: F401


#: features whose computation is fast
FEATURES_RAPID = [
    "area_ratio",
    "area_um",
    "aspect",
    "deform",
    "index",
    "time",
]


af_basic.register()
af_emodulus.register()
af_fl_max_ctc.register()
af_image_contour.register()
af_ml_class.register()
