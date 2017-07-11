#!/usr/bin/python
# -*- coding: utf-8 -*-
"""RT-DC dataset core classes and methods"""
from __future__ import division, print_function, unicode_literals

import abc

import numpy as np

from .. import definitions as dfn
from .. import downsampling
from ..polygon_filter import PolygonFilter
from .. import kde_methods

from .ancillary_columns import AncillaryColumn
from .export import Export
from .filter import Filter


class RTDCBase(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        """RT-DC measurement base class
        
        Notes
        -----
        Besides the filter arrays for each data column, there is a manual
        boolean filter array ``RTDCBase.filter.manual`` that can be edited
        by the user - a boolean value of ``False`` means that the event is 
        excluded from all computations.
        """
        # file format (derived from class name)
        self.format = self.__class__.__name__.split("_")[-1].lower()
        
        self._polygon_filter_ids = []
        # Ancillaries have the column name as keys and a
        # tuple containing column and hash as value.
        self._ancillaries = {}
        # export functionalities
        self.export = Export(self)


    def __contains__(self, key):
        ct = False
        if key in self._events:
            # Stored data contains events
            val = self._events[key]
            if isinstance(val, np.ndarray):
                # False of stored data is zero 
                ct = not np.all(val == 0)
            elif val:
                # True if stored data is not empty
                # (e.g. tdms image, trace, contour)
                ct = True
        if ct == False:
            # Check ancillary columns data
            if key in self._ancillaries:
                # already computed
                ct = True
            elif key in AncillaryColumn.column_names:
                cc = AncillaryColumn.get_column(key)
                if cc.is_available(self):
                    # to be computed
                    ct = True
        return ct


    def __getitem__(self, key):
        if key in self._events:
            data = self._events[key]
            if not np.all(data==0):
                return data 
        # Try to find the column in the ancillary columns
        # (see ancillary_columns.py for more information).
        # These columns are cached in `self._ancillaries`.
        ancol = AncillaryColumn.available_columns(self)
        if key in ancol:
            # The column is available.
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
            raise KeyError("Column '{}' does not exist!".format(key))


    def __iter__(self):
        """An iterator over all valid scalar columns"""
        mycols = []
        for col in dfn.column_names:
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
            raise ValueError("Could not determine size of data set.")
    
    
    def __repr__(self):
        repre = self.identifier
        if self.path is not "none":
            repre += " - file: {}".format(self.path)
        return repre
    
    
    @property
    def _filter(self):
        """return the current filter boolean array"""
        return self.filter.all


    def _init_filters(self):
        # Plot filters is only used for plotting and does
        # not have anything to do with filtering.
        self._plot_filter = np.ones(len(self), dtype=bool)
        
        self.filter = Filter(self)


    @property
    def identifier(self):
        """Compute an identifier based on __hash__"""
        return "mm-{}_{}".format(self.format, self.hash)


    def apply_filter(self, force=[]):
        """Computes the filters for the data set"""
        self.filter.update(force)


    @property
    def columns(self):
        """Return all available columns"""
        mycols = []
        for col in dfn.column_names + ["contour", "image", "trace"]:
            if col in self:
                mycols.append(col)
        mycols.sort()
        return mycols


    def get_downsampled_scatter(self, xax="area_um", yax="deform", downsample=0):
        """Downsampling by removing points at dense locations
        
        Parameters
        ----------
        xax: str
            Identifier for x axis (e.g. "area_um", "aspect", "deform", ...)
        yax: str
            Identifier for y axis
        downsample: int or None
            Number of points to draw in the down-sampled plot.
            This number is either 
            - >=1: exactly downsample to this number by randomly adding
                   or removing points 
            - 0  : do not perform downsampling

        Returns
        -------
        xnew, xnew: filtered x and y
        """
        assert downsample >= 0
        
        downsample = int(downsample)
        xax = xax.lower()
        yax = yax.lower()

        # Get axes
        if self.config["filtering"]["enable filters"]:
            x = self[xax][self._filter]
            y = self[yax][self._filter]
        else:
            # filtering disabled
            x = self[xax]
            y = self[yax]

        xsd, ysd, idx = downsampling.downsample_grid(x, y,
                                                     samples=downsample,
                                                     retidx=True)
        self._plot_filter = idx
        assert np.alltrue(x[idx] == xsd) 
        return xsd, ysd


    def get_kde_contour(self, xax="area_um", yax="deform", xacc=None, yacc=None,
                        kde_type="none", kde_kwargs={}):
        """Evaluate the kernel density estimate for contours
        
        
        Parameters
        ----------
        xax: str
            Identifier for X axis (e.g. "area_um", "aspect", "deform", ...)
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


        Returns
        -------
        X, Y, Z : coordinates
            The kernel density Z evaluated on a rectangular grid (X,Y).
        """
        xax = xax.lower()
        yax = yax.lower()
        kde_type = kde_type.lower()
        assert kde_type in kde_methods.methods

        if self.config["filtering"]["enable filters"]:
            x = self[xax][self._filter]
            y = self[yax][self._filter]
        else:
            x = self[xax]
            y = self[yax]
        
        # sensible default values
        cpstep = lambda a: (a.max()-a.min())/10
        if xacc is None:
            xacc = cpstep(x)
        if yacc is None:
            yacc = cpstep(x)

        # Ignore infs and nans
        bad = np.isinf(x)+np.isnan(x)+np.isinf(y)+np.isnan(y)
        xc = x[~bad]
        yc = y[~bad]
        xlin = np.arange(xc.min(), xc.max(), xacc)
        ylin = np.arange(yc.min(), yc.max(), yacc)
        xmesh, ymesh = np.meshgrid(xlin,ylin)

        kde_fct = kde_methods.methods[kde_type]
        if len(x):
            density = kde_fct(events_x=x, events_y=y,
                              xout=xmesh, yout=ymesh,
                              **kde_kwargs)
        else:
            density = []
            
        return xmesh, ymesh, density


    def get_kde_scatter(self, xax="area_um", yax="deform", positions=None,
                        kde_type="none", kde_kwargs={}):
        """Evaluate the kernel density estimate for scatter data

        
        Parameters
        ----------
        xax: str
            Identifier for X axis (e.g. "area_um", "aspect", "deform",...)
        yax: str
            Identifier for Y axis
        positions: list of points
            The positions where the KDE will be computed. Note that
            the KDE estimate is computed from the the points that
            are set in `self._filter`.
        kde_type: str
            The KDE method to use
        kde_kwargs: dict
            Additional keyword arguments to the KDE method 


        Returns
        -------
        density : 1d ndarray
            The kernel density evaluated for the filtered data points.
        """
        xax = xax.lower()
        yax = yax.lower()
        kde_type = kde_type.lower()
        assert kde_type in kde_methods.methods
        
        if self.config["filtering"]["enable filters"]:
            x = self[xax][self._filter]
            y = self[yax][self._filter]
        else:
            x = self[xax]
            y = self[yax]

        if positions is None:
            posx = None
            posy = None
        else:
            posx = positions[0]
            posy = positions[1]

        kde_fct = kde_methods.methods[kde_type]
        if len(x):
            density = kde_fct(events_x=x, events_y=y,
                              xout=posx, yout=posy,
                              **kde_kwargs)
        else:
            density = []
        
        return density


    @abc.abstractproperty
    def hash(self):
        """Hashing property must be defined by derived classes"""


    def polygon_filter_add(self, filt):
        """Associate a Polygon Filter with this instance
        
        Parameters
        ----------
        filt: int or instance of `PolygonFilter`
            The polygon filter to add
        """
        msg = "`filt` must be a number or instance of PolygonFilter."
        assert isinstance(filt, (PolygonFilter, int, float)), msg
        
        if isinstance(filt, PolygonFilter):
            uid=filt.unique_id
        else:
            uid=int(filt)
        # append item
        self.config["filtering"]["polygon filters"].append(uid)


    def polygon_filter_rm(self, filt):
        """Remove a polygon filter from this instance

        Parameters
        ----------
        filt: int or instance of `PolygonFilter`
            The polygon filter to remove
        """
        msg = "`filt` must be a number or instance of PolygonFilter."
        assert isinstance(filt, (PolygonFilter, int, float)), msg
        
        if isinstance(filt, PolygonFilter):
            uid = filt.unique_id
        else:
            uid = int(filt)
        # remove item
        self.config["filtering"]["polygon filters"].remove(uid)
