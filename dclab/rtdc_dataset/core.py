#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
RTDC_DataSet classes and methods
"""
from __future__ import division, print_function, unicode_literals

import hashlib
import sys
import warnings

import numpy as np

from .. import definitions as dfn
from .. import downsampling
from ..polygon_filter import PolygonFilter
from .. import kde_methods

from .export import Export


if sys.version_info[0] == 2:
    str_classes = (str, unicode)
else:
    str_classes = str


class RTDCBase(object):
    def __init__(self):
        """Base class for RT-DC data sets"""
        # Kernel density estimator dictionaries
        
        self._old_filters = {} # for comparison to new filters
        self._polygon_filter_ids = []
        # export functionalities
        self.export = Export(self)


    def __contains__(self, key):
        val = self[key]
        ct = False
        if isinstance(val, np.ndarray): 
            ct = np.all(val!=0)
        elif val:
            ct = True
        return ct


    def __getattr__(self, attr):
        # temporary workaround to dict-like behavior
        # This method will not be called unless __getattribute__ raises
        # an AttributeError.
        try: 
            data = self.__getitem__(attr)
        except:
            raise AttributeError("Column not found: {}".format(attr))
        else:
            warnings.warn("Using geattr to get columns is DEPRECATED!")
            return data


    def __getitem__(self, key):
        if key in self._events:
            data = self._events[key]
            if not np.all(data==0):
                return data
        # Compute new data columns dynamically
        dummy = np.zeros(len(self))
        if key == "area_um":
            pix = self.config["image"]["pix size"]
            return self["area"] * pix**2
        elif key == "time":
            if "framerate" in self.config:
                fr = self.config["framerate"]["frame rate"]
                return (self["frame"] - self["frame"][0]) / fr
            else:
                return dummy
        else:
            return dummy
        # TODO:
        # - add volume computation or elastic modulus computation here?


    def __len__(self):
        keys = list(self._events.keys())
        keys.sort()
        for kk in keys:
            length = len(self._events[kk])
            if length:
                return length
        else:
            raise ValueError("Could not determine size of data set.")
        
        
    def _init_filters(self):
        datalen = len(self)
        # Plotting filters, set by "get_downsampled_scatter".
        # This is a nested filter - it must be applied after self._filter
        # to get the plotted events.
        self._plot_filter = np.ones(datalen, dtype=bool)
        # Set array filters:
        # This is the filter that will be used for plotting:
        self._filter = np.ones(datalen, dtype=bool)
        # Manual filters, additionally defined by the user
        self._filter_manual = np.ones_like(self._filter)
        # Invalid filters
        self._filter_invalid = np.ones_like(self._filter)
        # Find attributes to be filtered
        # These are the filters from which self._filter is computed
        inifilter = np.ones(datalen, dtype=bool)
        for key in dfn.rdv:
            if key in self:
                # great, we are dealing with an array
                setattr(self, "_filter_"+key, inifilter.copy())
        self._filter_polygon = inifilter.copy()



    def ApplyFilter(self, force=[]):
        """ Computes the filters for the data set
        
        Uses `self._old_filters` to determine new filters.

        Parameters
        ----------
        force : list()
            A list of parameters that must be refiltered.


        Notes
        -----
        Updates `self.config["filtering"].
        
        The data is filtered according to filterable attributes in
        the global variable `dfn.cfgmap`.
        """

        # These lists may help us become very fast in the future
        newkeys = []
        oldvals = []
        newvals = []

        FIL = self.config["filtering"]

        ## Determine which data was updated
        OLD = self._old_filters
        
        for skey in list(FIL.keys()):
            if not skey in OLD:
                OLD[skey] = None
            if OLD[skey] != FIL[skey]:
                newkeys.append(skey)
                oldvals.append(OLD[skey])
                newvals.append(FIL[skey])

        # A simple filtering technique just updates the _filter_*
        # variables using the global dfn.cfgmap dictionary.
        
        # This line gets the attribute names of self that need updates.
        attr2update = []
        for k in newkeys:
            # k[:-4] because we want to crop " Min" and " Max"
            # "Polygon Filters" is not processed here.
            if k[:-4] in dfn.uid:
                attr2update.append(dfn.cfgmaprev[k[:-4]])

        for f in force:
            # Check if a correct variable is forced
            if f in list(dfn.cfgmaprev.keys()):
                attr2update.append(dfn.cfgmaprev[f])
            else:
                raise ValueError("Unknown variable {}".format(f))
        
        attr2update = np.unique(attr2update)

        for attr in attr2update:
            if attr in self:
                data = self[attr]
                fstart = dfn.cfgmap[attr]+" min"
                fend = dfn.cfgmap[attr]+" max"
                # If min and max exist and if they are not identical:
                indices = getattr(self, "_filter_"+attr)
                if (fstart in FIL and
                    fend in FIL and
                    FIL[fstart] != FIL[fend]):
                    # TODO: speedup
                    # Here one could check for smaller values in the
                    # lists oldvals/newvals that we defined above.
                    # Be sure to check against force in that case!
                    ivalstart = FIL[fstart]
                    ivalend = FIL[fend]
                    indices[:] = (ivalstart <= data)*(data <= ivalend)
                else:
                    indices[:] = True
        
        # Filter Polygons
        # check if something has changed
        pf_id = "polygon filters"
        if (
            (pf_id in FIL and not pf_id in OLD) or
            (pf_id in FIL and pf_id in OLD and
             FIL[pf_id] != OLD[pf_id])):
            self._filter_polygon[:] = True
            # perform polygon filtering
            for p in PolygonFilter.instances:
                if p.unique_id in FIL[pf_id]:
                    # update self._filter_polygon
                    # iterate through axes
                    datax = self[dfn.cfgmaprev[p.axes[0]]]
                    datay = self[dfn.cfgmaprev[p.axes[1]]]
                    self._filter_polygon *= p.filter(datax, datay)

        # Invalid filters
        self._filter_invalid[:] = True
        if FIL["remove invalid events"]:            
            for attr in dfn.cfgmap:
                if attr in self:
                    col = self[attr]
                    invalid = np.isinf(col)+np.isnan(col)
                    self._filter_invalid *= ~invalid


        # now update the entire object filter
        # get a list of all filters
        self._filter[:] = True

        if FIL["enable filters"]:
            for attr in dir(self):
                if attr.startswith("_filter_"):
                    self._filter[:] *= getattr(self, attr)
    
            # Filter with configuration keyword argument "Limit Events".
            # This additional step limits the total number of events in
            # self._filter.
            if FIL["limit events"] > 0:
                limit = FIL["limit events"]
                sub = self._filter[self._filter]
                _f, idx = downsampling.downsample_rand(sub,
                                                       samples=limit,
                                                       retidx=True)
                sub[~idx] = False
                self._filter[self._filter] = sub
        
        # Actual filtering is then done during plotting            
        self._old_filters = self.config.copy()["filtering"]


    def get_downsampled_scatter(self, xax="area", yax="defo", downsample=0):
        """Downsampling by removing points at dense locations
        
        Parameters
        ----------
        xax: str
            Identifier for x axis (e.g. "area", "area ratio","circ",...)
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
        self.ApplyFilter()
        
        downsample = int(downsample)
        xax = xax.lower()
        yax = yax.lower()

        # Get axes
        if self.config["filtering"]["enable filters"]:
            x = self[dfn.cfgmaprev[xax]][self._filter]
            y = self[dfn.cfgmaprev[yax]][self._filter]
        else:
            # filtering disabled
            x = self[dfn.cfgmaprev[xax]]
            y = self[dfn.cfgmaprev[yax]]

        xsd, ysd, idx = downsampling.downsample_grid(x, y,
                                                     samples=downsample,
                                                     retidx=True)
        self._plot_filter = idx
        assert np.alltrue(x[idx] == xsd) 
        return xsd, ysd


    def get_kde_contour(self, xax="area", yax="defo", xacc=None, yacc=None,
                        kde_type="none", kde_kwargs={}):
        """Evaluate the kernel density estimate for contours
        
        
        Parameters
        ----------
        xax: str
            Identifier for X axis (e.g. "Area", "Area Ratio","Circ",...)
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
            x = self[dfn.cfgmaprev[xax]][self._filter]
            y = self[dfn.cfgmaprev[yax]][self._filter]
        else:
            x = self[dfn.cfgmaprev[xax]]
            y = self[dfn.cfgmaprev[yax]]
        
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


    def get_kde_scatter(self, xax="area", yax="defo", positions=None,
                        kde_type="none", kde_kwargs={}):
        """Evaluate the kernel density estimate for scatter data

        
        Parameters
        ----------
        xax: str
            Identifier for X axis (e.g. "area", "area ratio","circ",...)
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
            x = self[dfn.cfgmaprev[xax]][self._filter]
            y = self[dfn.cfgmaprev[yax]][self._filter]
        else:
            x = self[dfn.cfgmaprev[xax]]
            y = self[dfn.cfgmaprev[yax]]

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


    def PolygonFilterAppend(self, filt):
        """Associates a Polygon Filter to the RTDC_DataSet
        
        filt can either be an integer or an instance of PolygonFilter
        """
        msg = "`filt` must be a number or instance of PolygonFilter."
        assert isinstance(filt, (PolygonFilter, int, float)), msg
        
        if isinstance(filt, PolygonFilter):
            uid=filt.unique_id
        else:
            uid=int(filt)
        # append item
        self.config["filtering"]["polygon filters"].append(uid)


    def PolygonFilterRemove(self, filt):
        """Opposite of PolygonFilterAppend """
        msg = "`filt` must be a number or instance of PolygonFilter."
        assert isinstance(filt, (PolygonFilter, int, float)), msg
        
        if isinstance(filt, PolygonFilter):
            uid = filt.unique_id
        else:
            uid = int(filt)
        # remove item
        self.config["filtering"]["polygon filters"].remove(uid)


def hashfile(fname, blocksize=65536):
    afile = open(fname, 'rb')
    hasher = hashlib.sha256()
    buf = afile.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read(blocksize)
    afile.close()
    return hasher.hexdigest()


def obj2str(obj):
    """Full string representation of an object for hashing"""
    if isinstance(obj, str_classes):
        return obj.encode("utf-8")
    elif isinstance(obj, (bool, int, float)):
        return str(obj).encode("utf-8")
    elif obj is None:
        return b"none"
    elif isinstance(obj, np.ndarray):
        return obj.tostring()
    elif isinstance(obj, tuple):
        return obj2str(list(obj))
    elif isinstance(obj, list):
        return b"".join(obj2str(o) for o in obj)
    elif isinstance(obj, dict):
        return obj2str(list(obj.items()))
    else:
        raise ValueError("No rule to convert object '{}' to string.".
                         format(obj.__class__))
