#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
RTDC_DataSet classes and methods
"""
from __future__ import division, print_function, unicode_literals

import hashlib
from nptdms import TdmsFile
import numpy as np
import os
import sys
import time
import warnings

from .. import config
from .. import definitions as dfn
from ..polygon_filter import PolygonFilter
from .. import kde_methods
from .config import Configuration
from .event_contour import ContourColumn
from .event_image import ImageColumn
from .event_trace import TraceColumn
from .export import Export


if sys.version_info[0] == 2:
    str_classes = (str, unicode)
else:
    str_classes = str


def DeprecationWarning(func, replacement=""):
    warnings.warn("{} IS DEPRECATED AND WILL BE REMOVED!".format(func.__name__))
    return func



class RTDC_DataSet(object):
    """ An RTDC measurement object.
    
    Parameters
    ----------
    tdms_path: str
        Path to a '.tdms' file. Only one of `tdms_path and `ddict` can
        be specified.
    ddict: dict
        Dictionary with keys from `dclab.definitions.uid` (e.g. "area", "defo")
        with which the class will be instantiated. Not '.tdms' file is required.
        The configuration is set to the default configuration fo dclab.
    
    Notes
    -----
    Besides the filter arrays for each data column, there is a manual
    boolean filter array ``RTDC_DataSet._filter_manual`` that can be edited
    by the user - a boolean value of ``False`` means that the event is 
    excluded from all computations.
    
    """
    def __init__(self, tdms_path=None, ddict=None, hparent=None):
        """ Load tdms file and set all variables """
        kwinput = [tdms_path, ddict, hparent].count(None)
        assert kwinput==2, "Specify tdms_path OR ddict OR hparent"

        # Kernel density estimator dictionaries
        self._old_filters = {} # for comparison to new filters
        self._Downsampled_Scatter = {}
        self._polygon_filter_ids = []
        
        if tdms_path is None:
            # We are given a dictionary with data values.
            # - create a unique fake title
            t = time.localtime()
            rand = "".join([ hex(r)[2:-1] for r in np.random.randint(
                                                                     10000,
                                                                     size=3)])
            tdms_path = "{}_{:02d}_{:02d}/{}.tdms".format(t[0],t[1],t[2],rand)

        # Initialize variables and generate hashes
        self.tdms_filename = tdms_path
        self.filename = tdms_path
        self.name = os.path.split(tdms_path)[1].split(".tdms")[0]
        self.fdir = os.path.dirname(tdms_path)
        mx = os.path.join(self.fdir, self.name.split("_")[0])
        self.title = u"{} - {}".format(
                                       GetProjectNameFromPath(tdms_path),
                                       os.path.split(mx)[1])
        fsh = [ tdms_path, mx+"_camera.ini", mx+"_para.ini" ]
        self.file_hashes = [(f, hashfile(f)) for f in fsh if os.path.exists(f)]
        ihasher = hashlib.md5()
        ihasher.update(obj2str(tdms_path))
        ihasher.update(obj2str(self.file_hashes))
        self.identifier = ihasher.hexdigest()

        if ddict is not None:
            # We are given a dictionary with data values.
            self._init_data_with_dict(ddict)
        elif hparent is not None:
            # We were given a hierarchy parent
            self._init_data_with_hierarchy(hparent)
        else:
            # We were given a tdms file.
            self._init_data_with_tdms(tdms_path)

        # event images
        self.image = ImageColumn(self)
        # event contours
        self.contour = ContourColumn(self)
        # event traces
        self.trace = TraceColumn(self)
        # compute other columns
        self.compute_columns()

        # export functionalities
        self.export = Export(self)


    def _init_data_with_dict(self, ddict):
        for key in ddict:
            setattr(self, dfn.cfgmaprev[key.lower()], np.array(ddict[key]))
        fill0 = np.zeros(len(ddict[key]))
        for key in dfn.rdv:
            if not hasattr(self, key):
                setattr(self, key, fill0)

        # Set up filtering
        self._init_filters()
        self.config = Configuration(rtdc_ds=self)


    def _init_data_with_hierarchy(self, hparent):
        """ Initializes the current RTDC_DataSet with another RTDC_Data set.
        
        A few words on hierarchies:
        The idea is that an RTDC_DataSet can use the filtered data of another
        RTDC_DataSet and interpret these data as unfiltered events. This comes
        in handy e.g. when the percentage of different subpopulations need to
        be distinguished without the noise in the original data.
        
        Children in hierarchies always update their data according to the
        filtered event data from their parent when `ApplyFilter` is called.
        This makes it easier to save and load hierarchy children with e.g.
        ShapeOut and it makes the handling of hierarchies more intuitive
        (when the parent changes, the child changes as well).
        
        Parameters
        ----------
        hparent : instance of RTDC_DataSet
            The hierarchy parent.
            
        Attributes
        ----------
        hparent : instance of RTDC_DataSet
            Only hierarchy children have this attribute
        """
        self.hparent = hparent
        
        ## Copy configuration
        cfg = hparent.config.copy()

        # Remove previously applied filters
        pops = []
        for key in cfg["Filtering"]:
            if (key.endswith("Min") or
                key.endswith("Max") or
                key == "Polygon Filters"):
                pops.append(key)
        [ cfg["Filtering"].pop(key) for key in pops ]
        # Add parent information in dictionary
        cfg["Filtering"]["Hierarchy Parent"]=hparent.identifier

        self.config = Configuration(cfg=cfg)

        myhash = hashlib.md5(obj2str(time.time())).hexdigest()
        self.identifier = hparent.identifier+"_child-"+myhash
        self.title = hparent.title + "_child-"+myhash[-4:]
        # Apply the filter
        # This will also populate all event attributes
        self.ApplyFilter()


    def _init_data_with_tdms(self, tdms_filename):
        """ Initializes the current RTDC_DataSet with a tdms file.
        """
        tdms_file = TdmsFile(tdms_filename)
        ## Set all necessary internal parameters as defined in
        ## definitions.py
        ## Note that this is meta-programming. If you want to add a
        ## different column from tdms files, then edit definitions.py:
        ## -> uid, axl, rdv, tfd
        # time is always there
        datalen = len(tdms_file.object("Cell Track", "time").data)
        for ii, group in enumerate(dfn.tfd):
            # ii iterates through the data that we could possibly extract
            # from a the tdms file.
            # The `group` contains all information necessary to extract
            # the data: table name, used column names, method to compute
            # the desired data from the columns.
            table = group[0]
            if not isinstance(group[1], list):
                # just for standards
                group[1] = [group[1]]
            func = group[2]
            args = []
            try:
                for arg in group[1]:
                    data = tdms_file.object(table, arg).data
                    if data is None or len(data)==0:
                        # Fill empty columns with zeros. npTDMS treats empty
                        # columns in the following way:
                        # - in nptdms 0.8.2, `data` is `None`
                        # - in nptdms 0.9.0, `data` is an array of length 0
                        data = np.zeros(datalen)
                    args.append(data)
            except KeyError:
                # set it to zero
                func = lambda x: x
                args = [np.zeros(datalen)]
            finally:
                setattr(self, dfn.rdv[ii], func(*args))


        # Set up filtering
        self._init_filters()
        mx = os.path.join(self.fdir, self.name.split("_")[0])
        self.config = Configuration(files=[mx+"_para.ini", mx+"_camera.ini"],
                                    rtdc_ds=self)
        self.SetConfiguration()


    def _init_filters(self):
        datalen = self.time.shape[0]
        # Plotting filters, set by "GetDownSampledScatter".
        # This is a nested filter which is applied after self._filter
        self._plot_filter = np.ones_like(self.time, dtype=bool)

        # Set array filters:
        # This is the filter that will be used for plotting:
        self._filter = np.ones_like(self.time, dtype=bool)
        # Manual filters, additionally defined by the user
        self._filter_manual = np.ones_like(self._filter)
        # The filtering array for a general data event limit:
        self._filter_limit = np.ones_like(self._filter)
        attrlist = dir(self)
        # Find attributes to be filtered
        # These are the filters from which self._filter is computed
        inifilter = np.ones(datalen, dtype=bool)
        for attr in attrlist:
            # only allow filterable attributes from global dfn.cfgmap
            if not attr in dfn.cfgmap:
                continue
            data = getattr(self, attr)
            if isinstance(data, np.ndarray):
                # great, we are dealing with an array
                setattr(self, "_filter_"+attr, inifilter.copy())
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
        Updates `self.config["Filtering"].
        
        The data is filtered according to filterable attributes in
        the global variable `dfn.cfgmap`.
        """

        # These lists may help us become very fast in the future
        newkeys = []
        oldvals = []
        newvals = []
        
        if not "filtering" in self.config:
            self.config["filtering"] = {"enable filters":False}

        FIL = self.config["filtering"]

        # Check if we are a hierarchy child and if yes, update the
        # filtered events from the hierarchy parent.
        if ("hierarchy parent" in FIL and
            FIL["hierarchy parent"].lower() != "none"):
            # Copy event data from hierarchy parent
            self.hparent.ApplyFilter()
            # TODO:
            # - somehow copy manually filtered events
            if (hasattr(self, "_filter_manual") 
                and np.sum(1-self._filter_manual) != 0):
                msg = "filter_manual not supported in hierarchy!"
                raise NotImplementedError(msg)

            for attr in dfn.rdv:
                filtevents = getattr(self.hparent, attr)[self.hparent._filter]
                setattr(self, attr, filtevents)
            self._init_filters()
            self._old_filters = {}

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
                warnings.warn(
                    "Unknown variable not force-filtered: {}".format(f))

        if "deform" in attr2update:
            attr2update.append("circ")
        elif "circ" in attr2update:
            attr2update.append("deform")
        
        attr2update = np.unique(attr2update)

        for attr in attr2update:
            data = getattr(self, attr)
            if isinstance(data, np.ndarray):
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
                    # Be sure to check agains force in that case!
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
                    datax = getattr(self, dfn.cfgmaprev[p.axes[0]])
                    datay = getattr(self, dfn.cfgmaprev[p.axes[1]])
                    self._filter_polygon *= p.filter(datax, datay)
        
        # Reset limit filters before
        # This is important. If we do not do this the we have
        # a pre-filter that does not make sense.
        self._filter_limit = np.ones_like(self._filter)
        
        # now update the entire object filter
        # get a list of all filters
        self._filter[:] = True
        if FIL["enable filters"]:
            for attr in dir(self):
                if attr.startswith("_filter_"):
                    self._filter[:] *= getattr(self, attr)
    
            # Filter with configuration keyword argument "Limit Events"
            if FIL["limit events"] > 0:
                limit = FIL["limit events"]
                incl = self._filter.copy()
                numevents = np.sum(incl)
                if limit < numevents:
                    # Perform equally distributed removal of events
                    # We have too many events
                    remove = int(numevents - limit)
                    while remove > 10:
                        there = np.where(incl)[0]
                        # first remove evenly distributed events
                        dist = int(np.ceil(there.shape[0]/remove))
                        incl[there[::dist]] = 0
                        numevents = np.sum(incl)
                        remove = int(numevents - limit)
                    there = np.where(incl)[0]
                    incl[there[:remove]] = 0
                    self._filter_limit = incl
                    print("'limit events' set to {}/{}".format(np.sum(incl), incl.shape[0]))
                elif limit == numevents:
                    # everything is ok
                    self._filter_limit = np.ones_like(self._filter)
                    print("'limit events' is size of filtered data.")
                else:
                    self._filter_limit = np.ones_like(self._filter)
                    warnings.warn("{}: 'Limit Events' must not ".format(self.name)+
                                  "be larger than length of data set! "+
                                  "Resetting 'Limit Events'!")
                    FIL["limit events"] = 0
            else:
                # revert everything back to how it was
                self._filter_limit = np.ones_like(self._filter)
            
            # Update filter again
            self._filter *= self._filter_limit

        # Actual filtering is then done during plotting            
        self._old_filters = self.config.copy()["filtering"]


    def compute_columns(self):
        """Compute columns that require information from self.config"""
        # TODO:
        # - create a list of definitions and compute the data later on
        if ("image" in self.config and
            "pix size" in self.config["image"]):
            PIX = self.config["image"]["pix size"]
            if not np.allclose(self.area, 0):
                self.area_um[:] = self.area * PIX**2
        # look for frame rate update
        if ("framerate" in self.config and
            "frame rate" in self.config["Framerate"]):
            FR = self.config["framerate"]["frame rate"]
            # FR is in Hz
            self.time[:] = (self.frame - self.frame[0]) / FR
        self.config._complete_config_from_rtdc_ds(self)


    @property
    def Configuration(self):
        warnings.warn("PLEASE USE RTDC_DataSet.config instead of RTDC_DataSet.Configuration!")
        return self.config


    @Configuration.setter
    def Configuration(self, k):
        warnings.warn("PLEASE DO NOT SET RTDC_DataSet.Configuration!")
        self.config = Configuration()
        self.config.update(k)


    def GetDownSampledScatter(self, c=None, axsize=(300,300),
                              markersize=1,
                              downsample_events=None):
        """ Filters a set of data from overlayed events for plotting
        
        Parameters
        ----------
        c : 1d array of same length as x and y
            Value (e.g. kernel density) for each point (x,y)
        axsize : size tuple
            Size of the axis.
        markersize : float
            Size of the marker (in dots), including edge.
        downsample_events : int or None
            Number of points to draw in the down-sampled plot.
            This number is either 
            - >=1: limit total number of events drawn
            - <1: only perform 1st downsampling step with grid
            If set to None, then
            self.config["Plotting"]["Downsample Events"]
            will be used.
        
        Returns
        -------
        xnew, xnew : filtered x and y
        """
        self.ApplyFilter()
        plotfilters = self.config["Plotting"]
        if downsample_events is None:
            downsample_events = plotfilters["Downsample Events"]
        downsample_events = int(downsample_events)

        downsampling = plotfilters["Downsampling"]            

        assert downsample_events > 0
        assert downsampling in [True, False]

        xax, yax = self.GetPlotAxes()

        # identifier for this setup
        hasher = hashlib.sha256()
        hasher.update(obj2str(axsize))
        hasher.update(obj2str(markersize))
        hasher.update(obj2str(c))
        # Get axes
        if self.config["filtering"]["enable filters"]:
            x = getattr(self, dfn.cfgmaprev[xax])[self._filter]
            y = getattr(self, dfn.cfgmaprev[yax])[self._filter]
            hasher.update(obj2str(self.config["filtering"]))
        else:
            # filtering disabled
            x = getattr(self, dfn.cfgmaprev[xax])
            y = getattr(self, dfn.cfgmaprev[yax])

        hasher.update(obj2str(downsample_events))
        hasher.update(obj2str(x))
        hasher.update(obj2str(y))
        hasher.update(obj2str(downsampling))
        identifier = hasher.hexdigest()

        if identifier in self._Downsampled_Scatter:
            result, self._plot_filter = self._Downsampled_Scatter[identifier]
            return result

        if (not downsampling or
           (downsample_events > 0 and downsample_events > x.shape[0])):
            # nothing to do
            self._plot_filter = np.ones_like(x, dtype=bool)
            if c is None:
                result = x, y
            else: 
                result = x, y, c
            return result

        # Create mask
        mask = np.ones(axsize, dtype=bool)
        
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
        
        xpx = (x-xmin)/(xmax-xmin) * axsize[0]
        ypx = (y-ymin)/(ymax-ymin) * axsize[1]
        
        gridx = np.linspace(0, axsize[0], axsize[0], endpoint=True).reshape(-1,1)
        gridy = np.linspace(0, axsize[1], axsize[1], endpoint=True).reshape(1,-1)
        
        incl = np.zeros_like(x, dtype=bool)
        
        a=time.time()
        # set values in mask to false as we iterate over x and y
        #R = markersize/2
        #Rsq = R**2
        D = markersize
        
        # Begin Downsampling
        eventmask = mask.copy()
        
        for i in range(len(x)):
            xi = xpx[i]
            yi = ypx[i]
            ## first filter for exactly overlapping events
            if not eventmask[int(xi-1), int(yi-1)]:
                continue
            eventmask[int(xi-1), int(yi-1)] = False
            #boolvals = (xi-gridx)**2 + (yi-gridy)**2 < Rsq
            ## second filter for multiple overlay
            boolvals = (np.abs(xi-gridx) < D) * (np.abs(yi-gridy) < D)
            if np.sum(mask[boolvals]) != 0:
                mask *= np.logical_not(boolvals)
                #mask = np.logical_and(np.logical_not(boolvals), mask)
                #mask[boolvals] = 0
                incl[i] = True
        print("downsample time:", time.time()-a)


        # Perform upsampling: include events to match downsample_events
        if downsample_events > 0:
            numevents = np.sum(incl)
            if downsample_events < numevents:
                # Perform equally distributed removal of events
                # We have too many events
                remove = numevents - downsample_events
                while remove > 10:
                    there = np.where(incl)[0]
                    # first remove evenly distributed events
                    dist = int(np.ceil(there.shape[0]/remove))
                    incl[there[::dist]] = 0
                    numevents = np.sum(incl)
                    remove = numevents - downsample_events
                there = np.where(incl)[0]
                incl[there[:remove]] = 0
            else:
                # Add equally distributed events in the case
                # where we have previously downsampled with a grid.
                # We have not enough events
                add = downsample_events - numevents
                while add > 10:
                    away = np.where(~incl)[0]
                    # first remove evenly distributed events
                    dist = int(np.ceil(away.shape[0]/add))
                    incl[away[::dist]] = 1
                    numevents = np.sum(incl)
                    add = downsample_events - numevents
                away = np.where(~incl)[0]
                incl[away[:add]] = 1

        xincl = x[incl]
        yincl = y[incl]

        if c is None:
            result = [xincl, yincl]
        else: 
            dens = c[np.where(incl)]
            result = [xincl, yincl, dens]
        
        self._Downsampled_Scatter[identifier] = result, incl
        self._plot_filter = incl

        return result


    def GetKDE_Contour(self, yax="defo", xax="area"):
        """ The evaluated Gaussian Kernel Density Estimate
        
        -> for contours
        
        
        Parameters
        ----------
        xax : str
            Identifier for X axis (e.g. "Area", "Area Ratio","Circ",...)
        yax : str
            Identifier for Y axis
        
        
        Returns
        -------
        X, Y, Z : coordinates
            The kernel density Z evaluated on a rectangular grid (X,Y).
        
        See Also
        --------
        `scipy.stats.gaussian_kde`
        `statsmodels.nonparametric.kernel_density.KDEMultivariate`
        """
        xax = xax.lower()
        yax = yax.lower()
        kde_type = self.config["plotting"]["kde"].lower()
        # dummy area-circ
        deltaarea = self.config["plotting"]["contour accuracy "+xax]
        deltacirc = self.config["plotting"]["contour accuracy "+yax]
        
        # setup
        if self.config["filtering"]["enable filters"]:
            x = getattr(self, dfn.cfgmaprev[xax])[self._filter]
            y = getattr(self, dfn.cfgmaprev[yax])[self._filter]
        else:
            x = getattr(self, dfn.cfgmaprev[xax])
            y = getattr(self, dfn.cfgmaprev[yax])

        # evaluation
        xlin = np.arange(x.min(), x.max(), deltaarea)
        ylin = np.arange(y.min(), y.max(), deltacirc)
        xmesh, ymesh = np.meshgrid(xlin,ylin)

        a = time.time()
        
        # Keyword arguments for kernel density estimation
        kde_kwargs = {
                      "events_x": x,
                      "events_y": y,
                      "xout": xmesh,
                      "yout": ymesh,
                      }
        
        if kde_type == "multivariate":
            bwx = self.config["plotting"]["kde multivariate "+xax]
            bwy = self.config["plotting"]["kde multivariate "+yax]
            kde_kwargs["bw"] = [bwx, bwy]
        
        kde_fct = getattr(kde_methods, "kde_"+kde_type)

        density = kde_fct(**kde_kwargs)

        print("KDE contour {} time: ".format(kde_type), time.time()-a)

        return xmesh, ymesh, density


    def GetKDE_Scatter(self, yax="defo", xax="area", positions=None):
        """ The evaluated Gaussian Kernel Density Estimate
        
        -> for scatter plots
        
        Parameters
        ----------
        xax : str
            Identifier for X axis (e.g. "Area", "Area Ratio","Circ",...)
        yax : str
            Identifier for Y axis
        positions : list of points
            The positions where the KDE will be computed. Note that
            the KDE estimate is computed from the the points that
            are set in `self._filter`.
        
        Returns
        -------
        density : 1d ndarray
            The kernel density evaluated for the filtered data points.
        
        
        See Also
        --------
        `RTDC_DataSet.ApplyFilter`
        `scipy.stats.gaussian_kde`
        `statsmodels.nonparametric.kernel_density.KDEMultivariate`
        
        """
        xax = xax.lower()
        yax = yax.lower()
        if self.config["filtering"]["enable filters"]:
            x = getattr(self, dfn.cfgmaprev[xax])[self._filter]
            y = getattr(self, dfn.cfgmaprev[yax])[self._filter]
        else:
            x = getattr(self, dfn.cfgmaprev[xax])
            y = getattr(self, dfn.cfgmaprev[yax])

        kde_type = self.config["plotting"]["kde"].lower()
        
        a = time.time()
        
        if positions is None:
            posx = None
            posy = None
        else:
            posx = positions[0]
            posy = positions[1]

        # Keyword arguments for kernel density estimation
        kde_kwargs = {
                      "events_x": x,
                      "events_y": y,
                      "xout": posx,
                      "yout": posy,
                      }
        
        if kde_type == "multivariate":
            bwx = self.config["plotting"]["kde multivariate "+xax]
            bwy = self.config["plotting"]["kde multivariate "+yax]
            kde_kwargs["bw"] = [bwx, bwy]
        
        kde_fct = getattr(kde_methods, "kde_"+kde_type)
        
        if len(x) != 0:
            density = kde_fct(**kde_kwargs)
        else:
            density = []
        
        print("KDE scatter {} time: ".format(kde_type), time.time()-a)
        return density


    @DeprecationWarning
    def GetPlotAxes(self):
        #return 
        p = self.config["Plotting"]
        if not "axis x" in p:
            p["axis x"] = "area"
        if not "axis y" in p:
            p["axis y"] = "defo"
        
        return [p["Axis X"].lower(), p["Axis Y"].lower()]


    def PolygonFilterAppend(self, filt):
        """ Associates a Polygon Filter to the RTDC_DataSet
        
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
        """ Opposite of PolygonFilterAppend """
        msg = "`filt` must be a number or instance of PolygonFilter."
        assert isinstance(filt, (PolygonFilter, int, float)), msg
        
        if isinstance(filt, PolygonFilter):
            uid = filt.unique_id
        else:
            uid = int(filt)
        # remove item
        self.config["filtering"]["polygon filters"].remove(uid)


    @DeprecationWarning
    def SetConfiguration(self):
        """ Import configuration of measurement
        
        Requires the files "MX_camera.ini" and "MX_para.ini" to be
        present in `self.fdir`. The string "MX_" is at the beginning of
        `self.name` (measurement identifier).
        
        This function is called during `__init__` and it is not
        necessary to run it twice.
        """
        self.config.update(config.cfg)
        for name, _hash in self.file_hashes:
            if name.endswith(".ini") and os.path.exists(name):
                newdict = config.load_config_file(name)
                self.config.update(newdict)


    @DeprecationWarning
    def UpdateConfiguration(self, newcfg):
        """ Update current configuration `self.config`.
        
        Parameters
        ----------
        newcfg : dict
            Dictionary to update `self.config`
        
        
        Returns
        -------
        None (dictionary is updated in place).
        
        
        Notes
        -----
        It is not required to update the entire configuration. Small
        changes can be made.
        """
        force = []
        # TODO:
        # - if this only applies to tmds files, put it in the corresponding init file
        #   "Pix Size" and "Framerate" should not be changed by the user. 
        # look for pixel size update first
        if ("Image" in newcfg and
            "Pix Size" in newcfg["Image"]):
            PIX = newcfg["Image"]["Pix Size"]
            if not np.allclose(self.area, 0):
                self.area_um[:] = self.area * PIX**2
                force.append("Area")
        # look for frame rate update
        if ("Framerate" in newcfg and
            "Frame Rate" in newcfg["Framerate"]):
            FR = newcfg["Framerate"]["Frame Rate"]
            # FR is in Hz
            self.time[:] = (self.frame - self.frame[0]) / FR
            force.append("Time")

        config.update_config_dict(self.config, newcfg)

        if "filtering" in newcfg:
            # Only writing the new Mins and Maxs is not enough
            # We need to also set the _filter_* attributes.
            self.ApplyFilter(force=force)
        
        # Reset additional information
        self.config["General"]["Cell Number"] = self.time.shape[0]



def hashfile(fname, blocksize=65536):
    afile = open(fname, 'rb')
    hasher = hashlib.sha256()
    buf = afile.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read(blocksize)
    afile.close()
    return hasher.hexdigest()


def GetProjectNameFromPath(path):
    """ Gets the project name from a path.
    
    For a path "/home/peter/hans/HLC12398/online/M1_13.tdms" or
    For a path "/home/peter/hans/HLC12398/online/data/M1_13.tdms" or
    without the ".tdms" file, this will return always "HLC12398".
    """
    if path.endswith(".tdms"):
        dirn = os.path.dirname(path)
    elif os.path.isdir(path):
        dirn = path
    else:
        dirn = os.path.dirname(path)
    # check if the directory contains data or is online
    root1, trail1 = os.path.split(dirn)
    root2, trail2 = os.path.split(root1)
    _root3, trail3 = os.path.split(root2)
    
    if trail1.lower() in ["online", "offline"]:
        # /home/peter/hans/HLC12398/online/
        project = trail2
    elif ( trail1.lower() == "data" and 
           trail2.lower() in ["online", "offline"] ):
        # this is olis new folder sctructure
        # /home/peter/hans/HLC12398/online/data/
        project = trail3
    else:
        if os.path.exists(path):
            warnings.warn("Non-standard directory naming scheme: {}".format(path))
        project = trail1
    return project


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
