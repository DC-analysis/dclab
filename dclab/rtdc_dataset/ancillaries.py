#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Computation of ancillary features

Ancillary features are computed on-the-fly in dclab if the
required data is available. The features are registered here
and are computed when `RTDCBase.__getitem__` is called with
the respective feature name. When `RTDCBase.__contains__` is
called with the feature name, then the feature is not yet
computed, but the prerequisites are evaluated:

In [1]: "emodulus" in rtdc_dataset  # nothing is computed
Out[1]: True
In [2]: rtdc_dataset["emodulus"]
Out[2]: ndarray([...])  # now data is computed and cached

Once the data has been computed, `RTDCBase` caches it in
the `_ancillaries` property dict together with a hash
that is computed with `AncillaryFeature.hash`. The hash
is computed from the feature data `req_features` and the
configuration metadata `req_config`.
"""
from __future__ import division, print_function, unicode_literals

import hashlib
import warnings

import numpy as np

from .. import features
from .util import obj2str


class AncillaryFeature():
    # Holds all instances of this class
    features = []
    feature_names = []
    def __init__(self, feature_name, method, req_config=[], req_features=[]):
        """A data feature that is computed from existing data
        
        Parameters
        ----------
        feature_name: str
            The name of the ancillary feature, e.g. "emodulus".
        method: callable
            The method that computes the feature. This method
            takes an instance of `RTDCBase` as argument.
        req_config: list
            Required configuration parameters to compute the feature,
            e.g. ["calculation", ["emodulus model", "emodulus viscosity"]]
        req_features: list
            Required existing features in the data set,
            e.g. ["area_cvx", "deform"]
        
        Notes
        -----
        `req_config` and `req_features` are used to test whether the
        feature can be computed in `self.is_available`.
        """
        self.feature_name = feature_name
        self.method = method
        self.req_config = req_config
        self.req_features = req_features
        
        # register this feature
        AncillaryFeature.features.append(self)
        AncillaryFeature.feature_names.append(feature_name)


    def __repr__(self):
        return "Ancillary feature: {}".format(self.feature_name)
   
    
    @staticmethod
    def available_features(rtdc_ds):
        """Determine available features for an RT-DS data set
        
        Parameters
        ----------
        rtdc_ds: instance of RTDCBase
            The data set to check availability for
        
        Returns
        -------
        features: dict
            Dictionary with feature names as keys and instances
            of `AncillaryFeature` as values.
        """
        cols = {}
        for inst in AncillaryFeature.features:
            if inst.is_available(rtdc_ds):
                cols[inst.feature_name] = inst
        return cols


    def compute(self, rtdc_ds):
        """Compute the feature with self.method

        Parameters
        ----------
        rtdc_ds: instance of RTDCBase
            The data set to compute the feature for
        
        Returns
        -------
        feature: array- or list-like
            The computed data feature (read-only).
        """
        data = self.method(rtdc_ds)
        dsize = len(rtdc_ds) - data.size

        if dsize > 0:
            msg = "Growing feature {} in {} by {} to match event number!"
            warnings.warn(msg.format(self.feature_name, rtdc_ds, abs(dsize)))
            data.resize(len(rtdc_ds), refcheck=False)
            data[-dsize:] = np.nan
        elif dsize < 0:
            msg = "Shrinking feature {} in {} by {} to match event number!"
            warnings.warn(msg.format(self.feature_name, rtdc_ds, abs(dsize)))
            data.resize(len(rtdc_ds), refcheck=False)

        data.setflags(write=False)
        return data


    @staticmethod
    def get_feature(feature_name):
        for col in AncillaryFeature.features:
            if col.feature_name == feature_name:
                return col
        else:
            raise KeyError("Feature {} not found.".format(feature_name))


    def hash(self, rtdc_ds):
        """Used for identifying an ancillary computation"""
        hasher = hashlib.md5()
        hasher.update(obj2str(self.req_config))
        for col in self.req_features:
            hasher.update(obj2str(rtdc_ds[col]))
        return hasher.hexdigest()


    def is_available(self, rtdc_ds, verbose=False):
        """Check whether the feature is available
        
        Parameters
        ----------
        rtdc_ds: instance of RTDCBase
            The data set to check availability for
        
        Returns
        -------
        available: bool
            `True`, if feature can be computed with `compute`
        """
        # Check config keys
        for item in self.req_config:
            section, keys = item
            if section not in rtdc_ds.config:
                if verbose:
                    print("{} not in config".format(section))
                return False
            else:
                for key in keys:
                    if key not in rtdc_ds.config[section]:
                        if verbose:
                            print("{} not in config['{}']".format(key,
                                                                  section))
                        return False
        # Check features
        for col in self.req_features:
            if col not in rtdc_ds:
                return False
        # All passed
        return True



def compute_area_ratio(mm):
    return mm["area_cvx"] / mm["area_msd"]

AncillaryFeature(feature_name="area_ratio",
                 method=compute_area_ratio,
                 req_features=["area_cvx", "area_msd"])


def compute_area_um(mm):
    pxs = mm.config["imaging"]["pixel size"]
    return mm["area_cvx"] * pxs**2

AncillaryFeature(feature_name="area_um",
                 method=compute_area_um,
                 req_config=[["imaging", ["pixel size"]]],
                 req_features=["area_cvx"])


def compute_aspect(mm):
    """Compute the aspect ratio of the bounding box
    
    Notes
    -----
    If the cell is elongated along the channel, i.e.
    `size_x` is larger than `size_y`, then the aspect
    ratio is larger than 1.
    """
    #parallel to flow, perpendicular to flow
    return mm["size_x"] / mm["size_y"]

AncillaryFeature(feature_name="aspect",
                 method=compute_aspect,
                 req_features=["size_x", "size_y"])


def compute_bright_avg(mm):
    bavg = features.bright.get_bright(cont=mm["contour"],
                                      img=mm["image"],
                                      ret_data="avg",
                                      )
    return bavg

AncillaryFeature(feature_name="bright_avg",
                 method=compute_bright_avg,
                 req_features=["image", "contour"])


def compute_bright_sd(mm):
    bstd = features.bright.get_bright(cont=mm["contour"],
                                      img=mm["image"],
                                      ret_data="sd",
                                      )
    return bstd

AncillaryFeature(feature_name="bright_sd",
                 method=compute_bright_sd,
                 req_features=["image", "contour"])


def compute_deform(mm):
    return 1 - mm["circ"]

AncillaryFeature(feature_name="deform",
                 method=compute_deform,
                 req_features=["circ"])


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
            area=mm["area_um"],
            deformation=mm["deform"],
            medium=medium,
            channel_width=mm.config["setup"]["channel width"],
            flow_rate=mm.config["setup"]["flow rate"],
            px_um=mm.config["imaging"]["pixel size"],
            temperature=mm.config["calculation"]["emodulus temperature"])
    return emod

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


def compute_fl1_max_ctc(mm):
    return features.fl_crosstalk.correct_crosstalk(
            fl1=mm["fl1_max"],
            fl2=mm["fl2_max"],
            fl3=mm["fl3_max"],
            fl_channel=1,
            ct21=mm.config["crosstalk fl21"],
            ct31=mm.config["crosstalk fl31"],
            ct12=mm.config["crosstalk fl12"],
            ct32=mm.config["crosstalk fl32"],
            ct13=mm.config["crosstalk fl13"],
            ct23=mm.config["crosstalk fl23"])

AncillaryFeature(feature_name="fl1_max_ctc",
                 method=compute_fl1_max_ctc,
                 req_features=["fl1_max", "fl2_max", "fl3_max"],
                 req_config=[["analysis", ["crosstalk fl21",
                                           "crosstalk fl31",
                                           "crosstalk fl12",
                                           "crosstalk fl32",
                                           "crosstalk fl13",
                                           "crosstalk fl23"]]
                             ])


def compute_fl2_max_ctc(mm):
    return features.fl_crosstalk.correct_crosstalk(
            fl1=mm["fl1_max"],
            fl2=mm["fl2_max"],
            fl3=mm["fl3_max"],
            fl_channel=2,
            ct21=mm.config["crosstalk fl21"],
            ct31=mm.config["crosstalk fl31"],
            ct12=mm.config["crosstalk fl12"],
            ct32=mm.config["crosstalk fl32"],
            ct13=mm.config["crosstalk fl13"],
            ct23=mm.config["crosstalk fl23"])

AncillaryFeature(feature_name="fl2_max_ctc",
                 method=compute_fl2_max_ctc,
                 req_features=["fl1_max", "fl2_max", "fl3_max"],
                 req_config=[["analysis", ["crosstalk fl21",
                                           "crosstalk fl31",
                                           "crosstalk fl12",
                                           "crosstalk fl32",
                                           "crosstalk fl13",
                                           "crosstalk fl23"]]
                             ])


def compute_fl3_max_ctc(mm):
    return features.fl_crosstalk.correct_crosstalk(
            fl1=mm["fl1_max"],
            fl2=mm["fl2_max"],
            fl3=mm["fl3_max"],
            fl_channel=3,
            ct21=mm.config["crosstalk fl21"],
            ct31=mm.config["crosstalk fl31"],
            ct12=mm.config["crosstalk fl12"],
            ct32=mm.config["crosstalk fl32"],
            ct13=mm.config["crosstalk fl13"],
            ct23=mm.config["crosstalk fl23"])

AncillaryFeature(feature_name="fl3_max_ctc",
                 method=compute_fl3_max_ctc,
                 req_features=["fl1_max", "fl2_max", "fl3_max"],
                 req_config=[["analysis", ["crosstalk fl21",
                                           "crosstalk fl31",
                                           "crosstalk fl12",
                                           "crosstalk fl32",
                                           "crosstalk fl13",
                                           "crosstalk fl23"]]
                             ])


def compute_index(mm):
    return np.arange(1, len(mm)+1)

AncillaryFeature(feature_name="index",
                 method=compute_index)


def compute_inert_ratio_cvx(mm):
    return features.inert_ratio.get_inert_ratio_cvx(cont=mm["contour"])

AncillaryFeature(feature_name="inert_ratio_cvx",
                 method=compute_inert_ratio_cvx,
                 req_features=["contour"])


def compute_inert_ratio_raw(mm):
    return features.inert_ratio.get_inert_ratio_raw(cont=mm["contour"])

AncillaryFeature(feature_name="inert_ratio_raw",
                 method=compute_inert_ratio_raw,
                 req_features=["contour"])


def compute_time(mm):
    fr = mm.config["imaging"]["frame rate"]
    return (mm["frame"] - mm["frame"][0]) / fr

AncillaryFeature(feature_name="time",
                 method=compute_time,
                 req_config=[["imaging", ["frame rate"]]],
                 req_features=["frame"])


def compute_volume(mm):
    vol = features.volume.get_volume(
                    cont=mm["contour"],
                    pos_x=mm["pos_x"],
                    pos_y=mm["pos_y"],
                    pix=mm.config["imaging"]["pixel size"])
    return vol

AncillaryFeature(feature_name="volume",
                 method=compute_volume,
                 req_features=["contour", "pos_x", "pos_y"],
                 req_config=[["imaging", ["pixel size"]]])
