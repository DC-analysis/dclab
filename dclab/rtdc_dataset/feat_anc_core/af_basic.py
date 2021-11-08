
import numpy as np
from .ancillary_feature import AncillaryFeature


def compute_area_ratio(mm):
    valid = mm["area_msd"] != 0
    out = np.nan * np.ones(len(mm), dtype=float)
    return np.divide(mm["area_cvx"], mm["area_msd"], where=valid, out=out)


def compute_area_um(mm):
    pxs = mm.config["imaging"]["pixel size"]
    return mm["area_cvx"] * pxs**2


def compute_aspect(mm):
    """Compute the aspect ratio of the bounding box

    Notes
    -----
    If the cell is elongated along the channel, i.e.
    `size_x` is larger than `size_y`, then the aspect
    ratio is larger than 1.
    """
    out = np.nan * np.ones(len(mm), dtype=float)
    valid = mm["size_y"] != 0
    # parallel to flow, perpendicular to flow
    return np.divide(mm["size_x"], mm["size_y"], where=valid, out=out)


def compute_deform(mm):
    return 1 - mm["circ"]


def compute_index(mm):
    return np.arange(1, len(mm)+1)


def compute_time(mm):
    fr = mm.config["imaging"]["frame rate"]
    return (mm["frame"] - mm["frame"][0]) / fr


AncillaryFeature(feature_name="time",
                 method=compute_time,
                 req_config=[["imaging", ["frame rate"]]],
                 req_features=["frame"])


AncillaryFeature(feature_name="index",
                 method=compute_index)


def register():
    AncillaryFeature(feature_name="area_ratio",
                     method=compute_area_ratio,
                     req_features=["area_cvx", "area_msd"])

    AncillaryFeature(feature_name="area_um",
                     method=compute_area_um,
                     req_config=[["imaging", ["pixel size"]]],
                     req_features=["area_cvx"])

    AncillaryFeature(feature_name="aspect",
                     method=compute_aspect,
                     req_features=["size_x", "size_y"])

    AncillaryFeature(feature_name="deform",
                     method=compute_deform,
                     req_features=["circ"])
