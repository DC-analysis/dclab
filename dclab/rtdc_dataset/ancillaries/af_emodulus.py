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


def compute_emodulus_visc_only(mm):
    """The user entered the viscosity directly

    This is actually a special case in :func:`compute_emodulus_legacy`.
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


def register():
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
                     priority=2)
    AncillaryFeature(feature_name="emodulus",
                     method=compute_emodulus_visc_only,
                     req_features=["area_um", "deform"],
                     req_config=[["calculation", ["emodulus model",
                                                  "emodulus viscosity"]],
                                 ["imaging", ["pixel size"]],
                                 ["setup", ["flow rate", "channel width"]]
                                 ],
                     priority=1)
    AncillaryFeature(feature_name="emodulus",
                     method=compute_emodulus_temp_feat,
                     req_features=["area_um", "deform", "temp"],
                     req_config=[["calculation", ["emodulus medium",
                                                  "emodulus model"]],
                                 ["imaging", ["pixel size"]],
                                 ["setup", ["flow rate", "channel width"]]
                                 ],
                     priority=0)
