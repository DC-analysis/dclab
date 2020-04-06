#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

from ... import features
from .ancillary_feature import AncillaryFeature


def compute_emodulus_legacy(mm):
    """This is how it was done in Shape-Out 1"""
    calccfg = mm.config["calculation"]
    model = calccfg["emodulus model"]
    assert model == "elastic sphere"
    medium = calccfg["emodulus medium"]
    viscosity = calccfg["emodulus viscosity"]
    temperature = mm.config["calculation"]["emodulus temperature"]
    if medium.lower() == "other":
        medium = viscosity
        temperature = None
    # compute elastic modulus
    emod = features.emodulus.get_emodulus(
        area_um=mm["area_um"],
        deform=mm["deform"],
        medium=medium,
        channel_width=mm.config["setup"]["channel width"],
        flow_rate=mm.config["setup"]["flow rate"],
        px_um=mm.config["imaging"]["pixel size"],
        temperature=temperature)
    return emod


def compute_emodulus_known_media(mm):
    """Only use known media and one temperature for all

    This is a special case in :func:`compute_emodulus_legacy`.
    """
    calccfg = mm.config["calculation"]
    model = calccfg["emodulus model"]
    assert model == "elastic sphere"
    medium = calccfg["emodulus medium"]
    if medium not in features.emodulus.viscosity.KNOWN_MEDIA:
        raise ValueError("Only the following media are supported: {}".format(
                         features.emodulus.viscosity.KNOWN_MEDIA))
    # compute elastic modulus
    emod = features.emodulus.get_emodulus(
        area_um=mm["area_um"],
        deform=mm["deform"],
        medium=medium,
        channel_width=mm.config["setup"]["channel width"],
        flow_rate=mm.config["setup"]["flow rate"],
        px_um=mm.config["imaging"]["pixel size"],
        temperature=mm.config["calculation"]["emodulus temperature"])
    return emod


def compute_emodulus_temp_feat(mm):
    """Use the "temperature" feature"""
    calccfg = mm.config["calculation"]
    model = calccfg["emodulus model"]
    assert model == "elastic sphere"
    medium = calccfg["emodulus medium"]
    assert medium != "other"
    # compute elastic modulus
    emod = features.emodulus.get_emodulus(
        area_um=mm["area_um"],
        deform=mm["deform"],
        medium=medium,
        channel_width=mm.config["setup"]["channel width"],
        flow_rate=mm.config["setup"]["flow rate"],
        px_um=mm.config["imaging"]["pixel size"],
        temperature=mm["temp"])
    return emod


def compute_emodulus_visc_only(mm):
    """The user entered the viscosity directly

    This is a special case in :func:`compute_emodulus_legacy`.
    """
    calccfg = mm.config["calculation"]
    model = calccfg["emodulus model"]
    assert model == "elastic sphere"
    viscosity = calccfg["emodulus viscosity"]
    # compute elastic modulus
    emod = features.emodulus.get_emodulus(
        area_um=mm["area_um"],
        deform=mm["deform"],
        medium=viscosity,
        channel_width=mm.config["setup"]["channel width"],
        flow_rate=mm.config["setup"]["flow rate"],
        px_um=mm.config["imaging"]["pixel size"],
        temperature=None)
    return emod


def is_channel(mm):
    """Check whether the measurement was performed in the channel

    If the chip region is not set, then it is assumed to be a
    channel measurement (for backwards compatibility and user-
    friendliness).
    """
    if "setup" in mm.config and "chip region" in mm.config["setup"]:
        region = mm.config["setup"]["chip region"]
        if region == "channel":
            # measured in the channel
            return True
        else:
            # measured in the reservoir
            return False
    else:
        # This might be a testing dictionary or someone who is
        # playing around with data. Avoid disappointments here.
        return True


def register():
    # Please note that registering these things is a delicate business,
    # because the priority has to be chosen carefully.
    AncillaryFeature(feature_name="emodulus",
                     method=compute_emodulus_legacy,
                     req_features=["area_um", "deform"],
                     req_config=[["calculation", ["emodulus medium",
                                                  "emodulus model",
                                                  "emodulus temperature",
                                                  "emodulus viscosity"]],
                                 ["imaging", ["pixel size"]],
                                 ["setup", ["flow rate", "channel width"]]
                                 ],
                     req_func=is_channel,
                     priority=3)
    AncillaryFeature(feature_name="emodulus",
                     method=compute_emodulus_known_media,
                     req_features=["area_um", "deform"],
                     req_config=[["calculation", ["emodulus medium",
                                                  "emodulus model",
                                                  "emodulus temperature"]],
                                 ["imaging", ["pixel size"]],
                                 ["setup", ["flow rate", "channel width"]]
                                 ],
                     req_func=is_channel,
                     priority=2)
    AncillaryFeature(feature_name="emodulus",
                     method=compute_emodulus_visc_only,
                     req_features=["area_um", "deform"],
                     req_config=[["calculation", ["emodulus model",
                                                  "emodulus viscosity"]],
                                 ["imaging", ["pixel size"]],
                                 ["setup", ["flow rate", "channel width"]]
                                 ],
                     req_func=is_channel,
                     priority=1)
    AncillaryFeature(feature_name="emodulus",
                     method=compute_emodulus_temp_feat,
                     req_features=["area_um", "deform", "temp"],
                     req_config=[["calculation", ["emodulus medium",
                                                  "emodulus model"]],
                                 ["imaging", ["pixel size"]],
                                 ["setup", ["flow rate", "channel width"]]
                                 ],
                     req_func=is_channel,
                     priority=0)
