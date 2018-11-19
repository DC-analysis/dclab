#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Isoelastics management

"""
from __future__ import division, unicode_literals

import pathlib
from pkg_resources import resource_filename

import numpy as np

from .. import definitions as dfn
from ..features import emodulus as feat_emod


ISOFILES = ["isoel-analytical-area_um-deform.txt",
            "isoel-numerical-area_um-deform.txt",
            ]
ISOFILES = [resource_filename("dclab.isoelastics", _if) for _if in ISOFILES]

VALID_METHODS = ["analytical", "numerical"]


class Isoelastics(object):
    def __init__(self, paths=[]):

        self._data = IsoelasticsDict()

        for path in paths:
            self.load_data(path)

    def _add(self, isoel, col1, col2, method, meta):
        """Convenience method for population self._data"""
        self._data[method][col1][col2]["isoelastics"] = isoel
        self._data[method][col1][col2]["meta"] = meta

        # Use advanced slicing to flip the data columns
        isoel_flip = [iso[:, [1, 0, 2]] for iso in isoel]
        self._data[method][col2][col1]["isoelastics"] = isoel_flip
        self._data[method][col2][col1]["meta"] = meta

    def add(self, isoel, col1, col2, channel_width,
            flow_rate, viscosity, method):
        """Add isoelastics

        Parameters
        ----------
        isoel: list of ndarrays
            Each list item resembles one isoelastic line stored
            as an array of shape (N,3). The last column contains
            the emodulus data.
        col1: str
            Name of the first feature of all isoelastics
            (e.g. isoel[0][:,0])
        col2: str
            Name of the second feature of all isoelastics
            (e.g. isoel[0][:,1])
        channel_width: float
            Channel width in µm
        flow_rate: float
            Flow rate through the channel in µl/s
        viscosity: float
            Viscosity of the medium in mPa*s
        method: str
            The method used to compute the isoelastics
            (must be one of `VALID_METHODS`).

        Notes
        -----
        The following isoelastics are automatically added for
        user convenience:
        - isoelastics with `col1` and `col2` interchanged
        - isoelastics for circularity if deformation was given
        """
        if method not in VALID_METHODS:
            validstr = ",".join(VALID_METHODS)
            raise ValueError("`method` must be one of {}!".format(validstr))
        for col in [col1, col2]:
            if col not in dfn.scalar_feature_names:
                raise ValueError("Not a valid feature name: {}".format(col))

        meta = [channel_width, flow_rate, viscosity]

        # Add the feature data
        self._add(isoel, col1, col2, method, meta)

        # Also add the feature data for circularity
        if "deform" in [col1, col2]:
            col1c, col2c = col1, col2
            if col1c == "deform":
                deform_ax = 0
                col1c = "circ"
            else:
                deform_ax = 1
                col2c = "circ"
            iso_circ = []
            for iso in isoel:
                iso = iso.copy()
                iso[:, deform_ax] = 1 - iso[:, deform_ax]
                iso_circ.append(iso)
            self._add(iso_circ, col1c, col2c, method, meta)

    @staticmethod
    def add_px_err(isoel, col1, col2, px_um, inplace=False):
        """Undo pixelation correction

        Isoelasticity lines are already corrected for pixelation
        effects as described in

        Mapping of Deformation to Apparent Young's Modulus
        in Real-Time Deformability Cytometry
        Christoph Herold, arXiv:1704.00572 [cond-mat.soft] (2017)
        https://arxiv.org/abs/1704.00572.

        If the isoealsticity lines are displayed with deformation data
        that are not corrected, then the lines must be "un"-corrected,
        i.e. the pixelation error must be added to the lines to match
        the experimental data.

        Parameters
        ----------
        isoel: list of 2d ndarrays of shape (N, 3)
            Each item in the list corresponds to one isoelasticity
            line. The first column is defined by `col1`, the second
            by `col2`, and the third column is the emodulus.
        col1, col2: str
            Define the fist to columns of each isoelasticity line.
            One of ["area_um", "circ", "deform"]
        px_um: float
            Pixel size [µm]
        """
        Isoelastics.check_col12(col1, col2)
        if "deform" in [col1, col2]:
            # add error for deformation
            sign = +1
        else:
            # subtract error for circularity
            sign = -1
        if col1 == "area_um":
            area_ax = 0
            deci_ax = 1
        else:
            area_ax = 1
            deci_ax = 0

        new_isoel = []
        for iso in isoel:
            iso = np.array(iso, copy=not inplace)
            ddeci = feat_emod.corrpix_deform_delta(area_um=iso[:, area_ax],
                                                   px_um=px_um)
            iso[:, deci_ax] += sign * ddeci
            new_isoel.append(iso)
        return new_isoel

    @staticmethod
    def check_col12(col1, col2):
        if (col1 not in ["area_um", "circ", "deform"] or
                col2 not in ["area_um", "circ", "deform"]):
            raise ValueError("Columns must be one of: area_um, circ, deform!")
        if col1 == col2:
            raise ValueError("Columns are the same!")
        if "area_um" not in [col1, col2]:
            # avoid [circ, deform]
            raise ValueError("One column must be set to 'area_um'!")

    @staticmethod
    def convert(isoel, col1, col2,
                channel_width_in, channel_width_out,
                flow_rate_in, flow_rate_out,
                viscosity_in, viscosity_out,
                inplace=False):
        """Convert isoelastics in area_um-deform space

        Parameters
        ----------
        isoel: list of 2d ndarrays of shape (N, 3)
            Each item in the list corresponds to one isoelasticity
            line. The first column is defined by `col1`, the second
            by `col2`, and the third column is the emodulus.
        col1, col2: str
            Define the fist to columns of each isoelasticity line.
            One of ["area_um", "circ", "deform"]
        channel_width_in: float
            Original channel width [µm]
        channel_width_out: float
            Target channel width [µm]
        flow_rate_in: float
            Original flow rate [µl/s]
        flow_rate_in: float
            Target flow rate [µl/s]
        viscosity_in: float
            Original viscosity [mPa*s]
        viscosity_out: float
            Target viscosity [mPa*s]

        Notes
        -----
        If only the positions of the isoelastics are of interest and
        not the value of the elastic modulus, then it is sufficient
        to supply values for the channel width and set the values
        for flow rate and viscosity to a constant (e.g. 1).

        See Also
        --------
        dclab.features.emodulus.convert: conversion method used
        """
        Isoelastics.check_col12(col1, col2)
        if col1 == "area_um":
            area_ax = 0
            defo_ax = 1
        else:
            area_ax = 1
            defo_ax = 0

        new_isoel = []

        for iso in isoel:
            iso = np.array(iso, copy=not inplace)
            feat_emod.convert(area_um=iso[:, area_ax],
                              deform=iso[:, defo_ax],
                              emodulus=iso[:, 2],
                              channel_width_in=channel_width_in,
                              channel_width_out=channel_width_out,
                              flow_rate_in=flow_rate_in,
                              flow_rate_out=flow_rate_out,
                              viscosity_in=viscosity_in,
                              viscosity_out=viscosity_out,
                              inplace=True)
            new_isoel.append(iso)
        return new_isoel

    def get(self, col1, col2, method, channel_width, flow_rate=None,
            viscosity=None, add_px_err=False, px_um=None):
        """Get isoelastics

        Parameters
        ----------
        col1: str
            Name of the first feature of all isoelastics
            (e.g. isoel[0][:,0])
        col2: str
            Name of the second feature of all isoelastics
            (e.g. isoel[0][:,1])
        method: str
            The method used to compute the isoelastics
            (must be one of `VALID_METHODS`).
        channel_width: float
            Channel width in µm
        flow_rate: float or `None`
            Flow rate through the channel in µl/s. If set to
            `None`, the flow rate of the imported data will
            be used (only do this if you do not need the
            correct values for elastic moduli).
        viscosity: float or `None`
            Viscosity of the medium in mPa*s. If set to
            `None`, the flow rate of the imported data will
            be used (only do this if you do not need the
            correct values for elastic moduli).
        add_px_err: bool
            If True, add pixelation errors according to
            C. Herold (2017), https://arxiv.org/abs/1704.00572
        px_um: float
            Pixel size [µm], used for pixelation error computation

        See Also
        --------
        dclab.features.emodulus.convert: conversion in-between
            channel sizes and viscosities
        dclab.features.emodulus.corrpix_deform_delta: pixelation
            error that is applied to the deformation data
        """
        if method not in VALID_METHODS:
            validstr = ",".join(VALID_METHODS)
            raise ValueError("`method` must be one of {}!".format(validstr))
        for col in [col1, col2]:
            if col not in dfn.scalar_feature_names:
                raise ValueError("Not a valid feature name: {}".format(col))

        if "isoelastics" not in self._data[method][col2][col1]:
            msg = "No isoelastics matching {}, {}, {}".format(col1, col2,
                                                              method)
            raise KeyError(msg)

        isoel = self._data[method][col1][col2]["isoelastics"]
        meta = self._data[method][col1][col2]["meta"]

        if flow_rate is None:
            flow_rate = meta[1]

        if viscosity is None:
            viscosity = meta[2]

        isoel_ret = self.convert(isoel, col1, col2,
                                 channel_width_in=meta[0],
                                 channel_width_out=channel_width,
                                 flow_rate_in=meta[1],
                                 flow_rate_out=flow_rate,
                                 viscosity_in=meta[2],
                                 viscosity_out=viscosity,
                                 inplace=False)

        if add_px_err:
            self.add_px_err(isoel=isoel_ret,
                            col1=col1,
                            col2=col2,
                            px_um=px_um,
                            inplace=True)

        return isoel_ret

    def get_with_rtdcbase(self, col1, col2, method, dataset,
                          viscosity=None, add_px_err=False):
        """Convenience method that extracts the metadata from RTDCBase

        Parameters
        ----------
        col1: str
            Name of the first feature of all isoelastics
            (e.g. isoel[0][:,0])
        col2: str
            Name of the second feature of all isoelastics
            (e.g. isoel[0][:,1])
        method: str
            The method used to compute the isoelastics
            (must be one of `VALID_METHODS`).
        dataset: dclab.rtdc_dataset.RTDCBase
            The dataset from which to obtain the metadata.
        viscosity: float or `None`
            Viscosity of the medium in mPa*s. If set to
            `None`, the flow rate of the imported data will
            be used (only do this if you do not need the
            correct values for elastic moduli).
        add_px_err: bool
            If True, add pixelation errors according to
            C. Herold (2017), https://arxiv.org/abs/1704.00572
        """
        cfg = dataset.config
        return self.get(col1=col1,
                        col2=col2,
                        method=method,
                        channel_width=cfg["setup"]["channel width"],
                        flow_rate=cfg["setup"]["flow rate"],
                        viscosity=viscosity,
                        add_px_err=add_px_err,
                        px_um=cfg["imaging"]["pixel size"])

    def load_data(self, path):
        """Load isoelastics from a text file

        The text file is loaded with `numpy.loadtxt` and must have
        three columns, representing the two data columns and the
        elastic modulus with units defined in `definitions.py`.
        The file header must have a section defining meta data of the
        content like so:

            # [...]
            #
            # - column 1: area_um
            # - column 2: deform
            # - column 3: emodulus
            # - channel width [um]: 20
            # - flow rate [ul/s]: 0.04
            # - viscosity [mPa*s]: 15
            # - method: analytical
            #
            # [...]


        Parameters
        ----------
        path: str
            Path to a isoelastics text file
        """
        path = pathlib.Path(path).resolve()
        # Get metadata
        meta = {}
        with path.open() as fd:
            while True:
                line = fd.readline().strip()
                if line.startswith("# - "):
                    line = line.strip("#- ")
                    var, val = line.split(":")
                    if val.strip().replace(".", "").isdigit():
                        # channel width, flow rate, viscosity
                        val = float(val)
                    else:
                        # columns, calculation
                        val = val.strip().lower()
                    meta[var.strip()] = val
                elif line and not line.startswith("#"):
                    break

        assert meta["column 1"] in dfn.scalar_feature_names
        assert meta["column 2"] in dfn.scalar_feature_names
        assert meta["column 3"] == "emodulus"
        assert meta["method"] in VALID_METHODS

        # Load isoelasics
        with path.open("rb") as isfd:
            isodata = np.loadtxt(isfd)

        # Slice out individual isoelastics
        emoduli = np.unique(isodata[:, 2])
        isoel = []
        for emod in emoduli:
            where = isodata[:, 2] == emod
            isoel.append(isodata[where])

        # Add isoelastics to instance
        self.add(isoel=isoel,
                 col1=meta["column 1"],
                 col2=meta["column 2"],
                 channel_width=meta["channel width [um]"],
                 flow_rate=meta["flow rate [ul/s]"],
                 viscosity=meta["viscosity [mPa*s]"],
                 method=meta["method"])


class IsoelasticsDict(dict):
    def __getitem__(self, key):
        if key in VALID_METHODS + dfn.scalar_feature_names:
            if key not in self:
                self[key] = IsoelasticsDict()
        return super(IsoelasticsDict, self).__getitem__(key)


def get_default():
    """Return default isoelasticity lines"""
    return Isoelastics(ISOFILES)
