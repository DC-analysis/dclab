#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Computation of ancillary columns

Ancillary columns are computed on-the-fly in dclab if the
required data is available. The columns are registered here
and are computed when `RTDCBase.__getitem__` is called with
the respective column name. When `RTDCBase.__contains__` is
called with the column name, then the column is not yet
computed, but the prerequisites are evaluated:

In [1]: "emodulus" in rtdc_dataset  # nothing is computed
Out[1]: True
In [2]: rtdc_dataset["emodulus"]
Out[2]: ndarray([...])  # now data is computed and cached

Once the data has been computed, `RTDCBase` caches it in
the `_ancillaries` property dict together with a hash
that is computed with `AncillaryColumn.hash`. The hash
is computed from the column data `req_columns` and the
configuration metadata `req_config`.
"""
from __future__ import division, print_function, unicode_literals

import hashlib

import numpy as np

from dclab import elastic, volume
from .util import obj2str


class AncillaryColumn():
    # Holds all instances of this class
    columns = []
    column_names = []
    def __init__(self, column_name, method, req_config=[], req_columns=[]):
        """A data column that is computed from existing data
        
        Parameters
        ----------
        column_name: str
            The name of the ancillary column, e.g. "emodulus".
        method: callable
            The method that computes the column. This method
            takes an instance of `RTDCBase` as argument.
        req_config: list
            Required configuration parameters to compute the column,
            e.g. ["calculation", ["emodulus model", "emodulus viscosity"]]
        req_columns: list
            Required existing columns in the data set,
            e.g. ["area", "defo"]
        
        Notes
        -----
        `req_config` and `req_columns` are used to test whether the
        column can be computed in `self.is_available`.
        """
        self.column_name = column_name
        self.method = method
        self.req_config = req_config
        self.req_columns = req_columns
        
        # register this column
        AncillaryColumn.columns.append(self)
        AncillaryColumn.column_names.append(column_name)


    def __repr__(self):
        return "Ancillary column: {}".format(self.column_name)


    def hash(self, rtdc_ds):
        """Used for identifying an ancillary computation"""
        hasher = hashlib.md5()
        hasher.update(obj2str(self.req_config))
        for col in self.req_columns:
            hasher.update(obj2str(rtdc_ds[col]))
        return hasher.hexdigest()
    
    
    @staticmethod
    def available_columns(rtdc_ds):
        """Determine available columns for an RT-DS data set
        
        Parameters
        ----------
        rtdc_ds: instance of RTDCBase
            The data set to check availability for
        
        Returns
        -------
        columns: dict
            Dictionary with column names as keys and instances
            of `AncillaryColumn` as values.
        """
        cols = {}
        for inst in AncillaryColumn.columns:
            if inst.is_available(rtdc_ds):
                cols[inst.column_name] = inst
        return cols


    @staticmethod
    def get_column(column_name):
        for col in AncillaryColumn.columns:
            if col.column_name == column_name:
                return col
        else:
            raise ValueError("Column {} not found.".format(column_name))


    def compute(self, rtdc_ds):
        """Compute the column with self.method

        Parameters
        ----------
        rtdc_ds: instance of RTDCBase
            The data set to compute the column for
        
        Returns
        -------
        column: array- or list-like
            The computed data column.
        """
        return self.method(rtdc_ds)


    def is_available(self, rtdc_ds, verbose=False):
        """Check whether the column is available
        
        Parameters
        ----------
        rtdc_ds: instance of RTDCBase
            The data set to check availability for
        
        Returns
        -------
        available: bool
            `True`, if column can be computed with `compute`
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
        # Check columns
        for col in self.req_columns:
            if col not in rtdc_ds:
                return False
        # All passed
        return True



def compute_area_ratio(mm):
    return mm["area"] / mm["area_raw"]


def compute_area_um(mm):
    pxs = mm.config["image"]["pix size"]
    return mm["area"] * pxs**2


def compute_aspect(mm):
    #perpendicular to flow, parallel to flow
    return mm["size_x"] / mm["size_y"]


def compute_deform(mm):
    return 1 - mm["circ"]


def compute_emodulus(mm):
    calccfg = mm.config["calculation"]

    model = calccfg["emodulus model"]
    assert model=="elastic sphere"
    
    medium = calccfg["emodulus medium"]
    viscosity = calccfg["emodulus viscosity"]
    if medium == "Other":
        medium = viscosity

    # compute elastic modulus
    emod = elastic.get_elasticity(
            area=mm["area_um"],
            deformation=mm["deform"],
            medium=medium,
            channel_width=mm.config["general"]["channel width"],
            flow_rate=mm.config["general"]["flow rate [ul/s]"],
            px_um=mm.config["image"]["pix size"],
            temperature=mm.config["calculation"]["emodulus temperature"])
    return emod


def compute_index(mm):
    return np.arange(1, len(mm)+1)


def compute_time(mm):
    fr = mm.config["framerate"]["frame rate"]
    return (mm["frame"] - mm["frame"][0]) / fr


def compute_volume(mm):
    vol = volume.get_volume(cont=mm["contour"],
                            pos_x=mm["pos_x"],
                            pos_y=mm["pos_y"],
                            pix=mm.config["image"]["pix size"])
    return vol
    

# Register ancillaries
AncillaryColumn(column_name="area_ratio",
                 method=compute_area_ratio,
                 req_columns=["area", "area_raw"]
                 )

AncillaryColumn(column_name="area_um",
                 method=compute_area_um,
                 req_config=[["image", ["pix size"]]],
                 req_columns=["area"]
                 )

AncillaryColumn(column_name="area_aspect",
                 method=compute_aspect,
                 req_columns=["size_x", "size_y"]
                 )

AncillaryColumn(column_name="deform",
                 method=compute_deform,
                 req_columns=["circ"]
                 )

# TODO:
# - Define multiple AncillaryColumn of "emodulus":
#   (e.g. using "temperature" column) 
AncillaryColumn(column_name="emodulus",
                 method=compute_emodulus,
                 req_columns=["area_um", "deform"],
                 req_config=[["calculation", 
                              ["emodulus medium",
                               "emodulus model",
                               "emodulus temperature",
                               "emodulus viscosity"]
                              ],
                             ["image",
                              ["pix size"]
                              ],
                             ["general", 
                              ["flow rate [ul/s]",
                               "channel width"]
                              ],
                             ],
                 )

AncillaryColumn(column_name="index",
                 method=compute_index,
                 )

AncillaryColumn(column_name="time",
                 method=compute_time,
                 req_config=[["framerate", ["frame rate"]]],
                 req_columns=["frame"]
                 )

AncillaryColumn(column_name="volume",
                 method=compute_volume,
                 req_columns=["contour", "pos_x", "pos_y"],
                 req_config=[["image", ["pix size"]]],
                 )
