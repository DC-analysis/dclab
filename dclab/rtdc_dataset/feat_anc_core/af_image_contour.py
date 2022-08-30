from ... import features
from .ancillary_feature import AncillaryFeature


def compute_contour(mm):
    cont = features.contour.get_contour_lazily(mask=mm["mask"])
    return cont


def compute_bright(mm):
    bavg, bsd = features.bright.get_bright(mask=mm["mask"],
                                           image=mm["image"],
                                           ret_data="avg,sd",
                                           )
    return {"bright_avg": bavg, "bright_sd": bsd}


def compute_bright_bc(mm):
    bavg, bsd = features.bright_bc.get_bright_bc(mask=mm["mask"],
                                                 image=mm["image"],
                                                 image_bg=mm["image_bg"],
                                                 ret_data="avg,sd",
                                                 )
    return {"bright_bc_avg": bavg, "bright_bc_sd": bsd}


def compute_bright_perc(mm):
    p10, p90 = features.bright_perc.get_bright_perc(mask=mm["mask"],
                                                    image=mm["image"],
                                                    image_bg=mm["image_bg"],
                                                    )
    return {"bright_perc_10": p10, "bright_perc_90": p90}


def compute_inert_ratio_cvx(mm):
    return features.inert_ratio.get_inert_ratio_cvx(cont=mm["contour"])


def compute_inert_ratio_prnc(mm):
    return features.inert_ratio.get_inert_ratio_prnc(cont=mm["contour"])


def compute_inert_ratio_raw(mm):
    return features.inert_ratio.get_inert_ratio_raw(cont=mm["contour"])


def compute_tilt(mm):
    return features.inert_ratio.get_tilt(cont=mm["contour"])


def compute_volume(mm):
    vol = features.volume.get_volume(
        cont=mm["contour"],
        pos_x=mm["pos_x"],
        pos_y=mm["pos_y"],
        pix=mm.config["imaging"]["pixel size"])
    return vol


def register():
    AncillaryFeature(feature_name="contour",
                     method=compute_contour,
                     req_features=["mask"])

    AncillaryFeature(feature_name="bright_avg",
                     method=compute_bright,
                     req_features=["image", "mask"])

    AncillaryFeature(feature_name="bright_sd",
                     method=compute_bright,
                     req_features=["image", "mask"])

    AncillaryFeature(feature_name="bright_bc_avg",
                     method=compute_bright_bc,
                     req_features=["image", "image_bg", "mask"])

    AncillaryFeature(feature_name="bright_bc_sd",
                     method=compute_bright_bc,
                     req_features=["image", "image_bg", "mask"])

    AncillaryFeature(feature_name="bright_perc_10",
                     method=compute_bright_perc,
                     req_features=["image", "image_bg", "mask"])

    AncillaryFeature(feature_name="bright_perc_90",
                     method=compute_bright_perc,
                     req_features=["image", "image_bg", "mask"])

    AncillaryFeature(feature_name="inert_ratio_cvx",
                     method=compute_inert_ratio_cvx,
                     req_features=["contour"])

    AncillaryFeature(feature_name="inert_ratio_prnc",
                     method=compute_inert_ratio_prnc,
                     req_features=["contour"])

    AncillaryFeature(feature_name="inert_ratio_raw",
                     method=compute_inert_ratio_raw,
                     req_features=["contour"])

    AncillaryFeature(feature_name="tilt",
                     method=compute_tilt,
                     req_features=["contour"])

    AncillaryFeature(feature_name="volume",
                     method=compute_volume,
                     req_features=["contour", "pos_x", "pos_y"],
                     req_config=[["imaging", ["pixel size"]]])
