#!/usr/bin/python
# -*- coding: utf-8 -*-
"""RT-DC dataset core classes and methods"""
from __future__ import division, print_function, unicode_literals

import abc
import random
import sys
import warnings

import numpy as np

from .. import definitions as dfn
from .. import downsampling
from ..polygon_filter import PolygonFilter
from .. import kde_methods

from .ancillaries import AncillaryFeature
from .export import Export
from .filter import Filter


class LogTransformWarning(UserWarning):
    pass


class RTDCBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, identifier=None):
        """RT-DC measurement base class

        Notes
        -----
        Besides the filter arrays for each data feature, there is a manual
        boolean filter array ``RTDCBase.filter.manual`` that can be edited
        by the user - a boolean value of ``False`` means that the event is
        excluded from all computations.
        """
        #: Dataset format (derived from class name)
        self.format = self.__class__.__name__.split("_")[-1].lower()

        self._polygon_filter_ids = []
        # Ancillaries have the feature name as keys and a
        # tuple containing feature and hash as value.
        self._ancillaries = {}
        #: Configuration of the measurement
        self.config = None
        #: Export functionalities; instance of
        #: :class:`dclab.rtdc_dataset.export.Export`.
        self.export = Export(self)
        # The filtering class is initialized with self._init_filters
        #: Filtering functionalities; instance of
        #: :class:`dclab.rtdc_dataset.filter.Filter`.
        self.filter = None
        #: Title of the measurement
        self.title = None
        # Unique identifier
        if identifier is None:
            # Generate a unique identifier for this dataset
            rhex = [random.choice('0123456789abcdef') for _n in range(7)]
            self._identifier = "mm-{}_{}".format(self.format, "".join(rhex))
        else:
            self._identifier = identifier

    def __contains__(self, key):
        ct = False
        if key in self._events:
            if (self.format == "tdms" and
                key in ["contour", "image", "trace"]
                    and self._events[key]):
                # Take into account special cases of the tdms file format:
                # tdms features "image", "trace", "contour" are True if
                # the data exist on disk
                ct = True
            else:
                ct = True
        if ct is False:
            # Check ancillary features data
            if key in self._ancillaries:
                # already computed
                ct = True
            elif key in AncillaryFeature.feature_names:
                # get all instance of AncillaryFeature that
                # compute the feature `key`
                instlist = AncillaryFeature.get_instances(key)
                for inst in instlist:
                    if inst.is_available(self):
                        # to be computed
                        ct = True
                        break
        return ct

    def __getitem__(self, key):
        if key in self._events:
            data = self._events[key]
            return data
        # Try to find the feature in the ancillary features
        # (see ancillaries submodule for more information).
        # These features are cached in `self._ancillaries`.
        ancol = AncillaryFeature.available_features(self)
        if key in ancol:
            # The feature is available.
            anhash = ancol[key].hash(self)
            if (key in self._ancillaries and
                    self._ancillaries[key][0] == anhash):
                # Use cached value
                data = self._ancillaries[key][1]
            else:
                # Compute new value
                data = ancol[key].compute(self)
                # Store computed value in `self._ancillaries`.
                self._ancillaries[key] = (anhash, data)
            return data
        else:
            raise KeyError("Feature '{}' does not exist!".format(key))

    def __iter__(self):
        """An iterator over all valid scalar features"""
        mycols = []
        for col in dfn.scalar_feature_names:
            if col in self:
                mycols.append(col)
        mycols.sort()
        for col in mycols:
            yield col

    def __len__(self):
        keys = list(self._events.keys())
        keys.sort()
        for kk in keys:
            length = len(self._events[kk])
            if length:
                return length
        else:
            msg = "Could not determine size of dataset '{}'.".format(self)
            raise ValueError(msg)

    def __repr__(self):
        repre = self.identifier
        if self.path != "none":
            if sys.version_info[0] == 2:
                repre += " - file: {}".format(str(self.path).decode("utf-8"))
            else:
                repre += " - file: {}".format(self.path)
        return repre

    def _apply_scale(self, a, scale, feat):
        """Helper function for transforming an aray to log-scale

        Parameters
        ----------
        a: np.ndarray
            Input array
        scale:
            If set to "log", take the logarithm of `a`; if set to
            "linear" return `a` unchanged.

        Returns
        -------
        b: np.ndarray
            The scaled array

        Notes
        -----
        If the scale is not "linear", then a new array is returned.
        All warnings are suppressed when computing `np.log(a)`, as
        `a` may have negative or nan values.
        """
        if scale == "linear":
            b = a
        elif scale == "log":
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                b = np.log(a)
                if len(w):
                    # Tell the user that the log-transformation issued
                    # a warning.
                    warnings.warn("Invalid values encounterd in np.log "
                                  "while scaling feature '{}'!".format(feat))
        else:
            raise ValueError("`scale` must be either 'linear' or 'log', "
                             + "got '{}'!".format(scale))
        return b

    @property
    def _filter(self):
        """return the current filter boolean array"""
        return self.filter.all

    def _init_filters(self):
        # Plot filters is only used for plotting and does
        # not have anything to do with filtering.
        self._plot_filter = np.ones(len(self), dtype=bool)

        #: Filtering functionalities (this is an instance of
        #: :class:`dclab.rtdc_dataset.filter.Filter`.
        self.filter = Filter(self)

    @property
    def identifier(self):
        """Unique (unreproducible) identifier"""
        return self._identifier

    @property
    def features(self):
        """All available features"""
        mycols = []
        for col in dfn.feature_names:
            if col in self:
                mycols.append(col)
        mycols.sort()
        return mycols

    @abc.abstractproperty
    def hash(self):
        """Reproducible dataset hash (defined by derived classes)"""

    def apply_filter(self, force=[]):
        """Compute the filters for the dataset"""
        self.filter.update(force)

    def get_downsampled_scatter(self, xax="area_um", yax="deform",
                                downsample=0, xscale="linear",
                                yscale="linear"):
        """Downsampling by removing points at dense locations

        Parameters
        ----------
        xax: str
            Identifier for x axis (e.g. "area_um", "aspect", "deform")
        yax: str
            Identifier for y axis
        downsample: int
            Number of points to draw in the down-sampled plot.
            This number is either

            - >=1: exactly downsample to this number by randomly adding
                   or removing points
            - 0  : do not perform downsampling
        xscale: str
            If set to "log", take the logarithm of the x-values before
            performing downsampling. This is useful when data are are
            displayed on a log-scale. Defaults to "linear".
        yscale: str
            See `xscale`.

        Returns
        -------
        xnew, xnew: filtered x and y
        """
        if downsample < 0:
            raise ValueError("`downsample` must be zero or positive!")

        downsample = int(downsample)
        xax = xax.lower()
        yax = yax.lower()

        # Get data
        x = self[xax][self.filter.all]
        y = self[yax][self.filter.all]

        # Apply scale (no change for linear scale)
        xs = self._apply_scale(x, xscale, xax)
        ys = self._apply_scale(y, yscale, yax)

        _, _, idx = downsampling.downsample_grid(xs, ys,
                                                 samples=downsample,
                                                 ret_idx=True)
        self._plot_filter = idx
        return x[idx], y[idx]

    def get_kde_contour(self, xax="area_um", yax="deform", xacc=None,
                        yacc=None, kde_type="histogram", kde_kwargs={},
                        xscale="linear", yscale="linear"):
        """Evaluate the kernel density estimate for contour plots

        Parameters
        ----------
        xax: str
            Identifier for X axis (e.g. "area_um", "aspect", "deform")
        yax: str
            Identifier for Y axis
        xacc: float
            Contour accuracy in x direction
        yacc: float
            Contour accuracy in y direction
        kde_type: str
            The KDE method to use
        kde_kwargs: dict
            Additional keyword arguments to the KDE method
        xscale: str
            If set to "log", take the logarithm of the x-values before
            computing the KDE. This is useful when data are are
            displayed on a log-scale. Defaults to "linear".
        yscale: str
            See `xscale`.

        Returns
        -------
        X, Y, Z : coordinates
            The kernel density Z evaluated on a rectangular grid (X,Y).
        """
        xax = xax.lower()
        yax = yax.lower()
        kde_type = kde_type.lower()
        if kde_type not in kde_methods.methods:
            raise ValueError("Not a valid kde type: {}!".format(kde_type))

        # Get data
        x = self[xax][self.filter.all]
        y = self[yax][self.filter.all]

        # Apply scale (no change for linear scale)
        xs = self._apply_scale(x, xscale, xax)
        ys = self._apply_scale(y, yscale, yax)

        # accuracy (bin width) of KDE estimator
        if xacc is None:
            xacc = kde_methods.bin_width_doane(xs) / 5
        if yacc is None:
            yacc = kde_methods.bin_width_doane(ys) / 5

        # Ignore infs and nans
        bad = kde_methods.get_bad_vals(xs, ys)
        xc = xs[~bad]
        yc = ys[~bad]

        xnum = int(np.ceil((xc.max() - xc.min()) / xacc))
        ynum = int(np.ceil((yc.max() - yc.min()) / yacc))

        xlin = np.linspace(xc.min(), xc.max(), xnum, endpoint=True)
        ylin = np.linspace(yc.min(), yc.max(), ynum, endpoint=True)

        xmesh, ymesh = np.meshgrid(xlin, ylin, indexing="ij")

        kde_fct = kde_methods.methods[kde_type]
        if len(x):
            density = kde_fct(events_x=xs, events_y=ys,
                              xout=xmesh, yout=ymesh,
                              **kde_kwargs)
        else:
            density = []

        # Convert mesh back to linear scale if applicable
        if xscale == "log":
            xmesh = np.exp(xmesh)
        if yscale == "log":
            ymesh = np.exp(ymesh)

        return xmesh, ymesh, density

    def get_kde_scatter(self, xax="area_um", yax="deform", positions=None,
                        kde_type="histogram", kde_kwargs={}, xscale="linear",
                        yscale="linear"):
        """Evaluate the kernel density estimate for scatter plots

        Parameters
        ----------
        xax: str
            Identifier for X axis (e.g. "area_um", "aspect", "deform")
        yax: str
            Identifier for Y axis
        positions: list of two 1d ndarrays or ndarray of shape (2, N)
            The positions where the KDE will be computed. Note that
            the KDE estimate is computed from the the points that
            are set in `self.filter.all`.
        kde_type: str
            The KDE method to use
        kde_kwargs: dict
            Additional keyword arguments to the KDE method
        xscale: str
            If set to "log", take the logarithm of the x-values before
            computing the KDE. This is useful when data are are
            displayed on a log-scale. Defaults to "linear".
        yscale: str
            See `xscale`.

        Returns
        -------
        density : 1d ndarray
            The kernel density evaluated for the filtered data points.
        """
        xax = xax.lower()
        yax = yax.lower()
        kde_type = kde_type.lower()
        if kde_type not in kde_methods.methods:
            raise ValueError("Not a valid kde type: {}!".format(kde_type))

        # Get data
        x = self[xax][self.filter.all]
        y = self[yax][self.filter.all]

        # Apply scale (no change for linear scale)
        xs = self._apply_scale(x, xscale, xax)
        ys = self._apply_scale(y, yscale, yax)

        if positions is None:
            posx = None
            posy = None
        else:
            posx = self._apply_scale(positions[0], xscale, xax)
            posy = self._apply_scale(positions[1], yscale, yax)

        kde_fct = kde_methods.methods[kde_type]
        if len(x):
            density = kde_fct(events_x=xs, events_y=ys,
                              xout=posx, yout=posy,
                              **kde_kwargs)
        else:
            density = []

        return density

    def polygon_filter_add(self, filt):
        """Associate a Polygon Filter with this instance

        Parameters
        ----------
        filt: int or instance of `PolygonFilter`
            The polygon filter to add
        """
        if not isinstance(filt, (PolygonFilter, int, float)):
            msg = "`filt` must be a number or instance of PolygonFilter!"
            raise ValueError(msg)

        if isinstance(filt, PolygonFilter):
            uid = filt.unique_id
        else:
            uid = int(filt)
        # append item
        self.config["filtering"]["polygon filters"].append(uid)

    def polygon_filter_rm(self, filt):
        """Remove a polygon filter from this instance

        Parameters
        ----------
        filt: int or instance of `PolygonFilter`
            The polygon filter to remove
        """
        if not isinstance(filt, (PolygonFilter, int, float)):
            msg = "`filt` must be a number or instance of PolygonFilter!"
            raise ValueError(msg)

        if isinstance(filt, PolygonFilter):
            uid = filt.unique_id
        else:
            uid = int(filt)
        # remove item
        self.config["filtering"]["polygon filters"].remove(uid)
