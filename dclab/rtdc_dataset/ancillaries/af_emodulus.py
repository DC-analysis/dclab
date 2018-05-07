#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

from ... import features
from .ancillary_feature import AncillaryFeature


def compute_emodulus(mm):
    calccfg = mm.config["calculation"]
    model = calccfg["emodulus model"]
    assert model == "elastic sphere"
    medium = calccfg["emodulus medium"]
    viscosity = calccfg["emodulus viscosity"]
    if medium == "Other":
        medium = viscosity
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


def register():
    # TODO:
    # - Define multiple AncillaryFeature of "emodulus":
    #   (e.g. using "temperature" feature)
    AncillaryFeature(feature_name="emodulus",
                     method=compute_emodulus,
                     req_features=["area_um", "deform"],
                     req_config=[["calculation", ["emodulus medium",
                                                  "emodulus model",
                                                  "emodulus temperature",
                                                  "emodulus viscosity"]],
                                 ["imaging", ["pixel size"]],
                                 ["setup", ["flow rate", "channel width"]]
                                 ])
