#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Isoelastics management"""
from __future__ import division, unicode_literals

import io
from pkg_resources import resource_filename

import numpy as np

from .. import definitions as dfn

ISOFILES = ["isoel-analytical-area_um-deform.txt"]
ISOFILES = [resource_filename("dclab.isoelastics", _if) for _if in ISOFILES]

VALID_METHODS = ["analytical", "fem"]


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
            if col not in dfn.feature_names:
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
    def convert(isoel, col1, col2,
                channel_width_in, channel_width_out,
                flow_rate_in, flow_rate_out,
                viscosity_in, viscosity_out):
        """Convert isoelastics in area_um-deform space

        The conversion formula is described in

            Extracting Cell Stiffness from Real-Time Deformability
            Cytometry: Theory and Experiment
            A. Mietke, O. Otto, S. Girardo, P. Rosendahl,
            A. Taubenberger, S. Golfier, E. Ulbricht,
            S. Aland, J. Guck, E. Fischer-Friedrich
            Biophysical Journal 109(10) 2015
            DOI: 10.1016/j.bpj.2015.09.006


        Notes
        -----
        If only the positions of the isoelastics are of interest and
        not the value of the elastic modulus, then it is sufficient
        to supply values for the channel width and set the values
        for flow rate and viscosity to a constant (e.g. 1).
        """
        if (col1 not in ["area_um", "circ", "deform"] or
                col2 not in ["area_um", "circ", "deform"]):
            raise ValueError("Columns must be one of: area_um, circ, deform!")
        if col1 == col2:
            raise ValueError("Columns are the same!")
        if "area_um" not in [col1, col2]:
            # avoid [circ, deform]
            raise ValueError("One column must be set to 'area_um'!")

        if col1 == "area_um":
            area_ax = 0
        else:
            area_ax = 1

        if (channel_width_in == channel_width_out and
            viscosity_in == viscosity_out and
                flow_rate_in == flow_rate_out):
            # Nothing to do
            return isoel

        new_isoel = []

        for iso in isoel:
            iso = iso.copy()
            if channel_width_in != channel_width_out:
                # convert lut area axis to match channel width
                iso[:, area_ax] *= (channel_width_out / channel_width_in)**2

            iso[:, 2] *= (flow_rate_out / flow_rate_in) *\
                (viscosity_out / viscosity_in) *\
                (channel_width_in / channel_width_out)**3

            new_isoel.append(iso)
        return new_isoel

    def get(self, col1, col2, method, channel_width,
            flow_rate=None, viscosity=None):
        """Get isoelastics

        Parameters
        ----------
        col1: str
            Name of the first feature of all isoelastics
            (e.g. isoel[0][:,0])
        col2: str
            Name of the second feature of all isoelastics
            (e.g. isoel[0][:,1])
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
        method: str
            The method used to compute the isoelastics
            (must be one of `VALID_METHODS`).
        """
        if method not in VALID_METHODS:
            validstr = ",".join(VALID_METHODS)
            raise ValueError("`method` must be one of {}!".format(validstr))
        for col in [col1, col2]:
            if col not in dfn.feature_names:
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
                                 viscosity_out=viscosity)
        return isoel_ret

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
        # Get metadata
        meta = {}
        with io.open(str(path)) as fd:
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

        assert meta["column 1"] in dfn.feature_names
        assert meta["column 2"] in dfn.feature_names
        assert meta["column 3"] == "emodulus"
        assert meta["method"] in VALID_METHODS

        # Load isoelasics
        isodata = np.loadtxt(str(path))

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
        if key in VALID_METHODS + dfn.feature_names:
            if key not in self:
                self[key] = IsoelasticsDict()
        return super(IsoelasticsDict, self).__getitem__(key)


default = Isoelastics(ISOFILES)
