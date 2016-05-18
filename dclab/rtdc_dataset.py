#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
RTDC_DataSet classes and methods
"""
from __future__ import division, print_function, unicode_literals

import codecs
import copy
import hashlib
from nptdms import TdmsFile
import numpy as np
import os
import time
    
import warnings

# Definitions
from . import definitions as dfn
from .polygon_filter import PolygonFilter
from . import kde_methods

class RTDC_DataSet(object):
    """ An RTDC measurement object.
    
    The object must be initiated with a '.tdms' filename.
    
    Notes
    -----
    Besides the filter arrays for each data column, there is a manual
    boolean filter array ``RTDC_DataSet._filter_manual`` that can be edited
    by the user - a boolean value of ``False`` means that the event is 
    excluded from all computations.
    
    """
    def __init__(self, tdms_filename):
        """ Load tdms file and set all variables """
        # Kernel density estimator dictionaries
        self._old_filters = {} # for comparison to new filters
        self._Downsampled_Scatter = {}
        self._polygon_filter_ids = []
        
        self.tdms_filename = tdms_filename
        self.name = os.path.split(tdms_filename)[1].split(".tdms")[0]
        self.fdir = os.path.dirname(tdms_filename)

        mx = os.path.join(self.fdir, self.name.split("_")[0])
        
        self.title = u"{} - {}".format(
                      GetProjectNameFromPath(tdms_filename),
                      os.path.split(mx)[1])
        
        f2hash = [ tdms_filename, mx+"_camera.ini", mx+"_para.ini" ]
        
        self.file_hashes = [(fname, _hashfile(fname)) for fname in f2hash]

        self.identifier = self.file_hashes[0][1]

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
                    if data is None:
                        # Sometimes the column is empty. Fill it
                        # with zeros:
                        data = np.zeros(datalen)
                    args.append(data)
            except KeyError:
                # set it to zero
                func = lambda x: x
                args = [np.zeros(datalen)]
            finally:
                setattr(self, dfn.rdv[ii], func(*args))

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

        self.SetConfiguration()

        # Get video file name
        videos = [v for v in os.listdir(self.fdir) if v.endswith(".avi")]
        # Filter videos according to measurement number
        meas_id = self.name.split("_")[0]
        videos = [v for v in videos if v.split("_")[0] == meas_id] 

        videos.sort()
        if len(videos) == 0:
            self.video = None
        else:
            # Defaults to first avi file
            self.video = videos[0]
            # g/q video file names. q comes first.
            for v in videos:
                if v.endswith("imag.avi"):
                    self.video = v
                    break
                # add this here, because fRT-DC measurements also contain
                # videos ..._proc.avi
                elif v.endswith("imaq.avi"):
                    self.video = v
                    break
        
        # Get contours
        self.contours = {}
        for f in os.listdir(self.fdir):
            if f.endswith("_contours.txt") and f.startswith(self.name[:2]):
                with open(os.path.join(self.fdir, f), "r") as c:
                    # read entire file
                    cdat = c.read(-1)
                for cont in cdat.split("Contour in frame"):
                    cont = cont.strip()
                    if len(cont) == 0:
                        continue
                    cont = cont.splitlines()
                    # the frame is the first number
                    frame = int(cont.pop(0))
                    cont = [ np.fromstring(c.strip("()"), sep=",") for c in cont ]
                    cont = np.array(cont, dtype=np.uint8)
                    self.contours[frame] = cont

        # Get traces
        self.traces = {}
        traces_filename = tdms_filename[:-5]+"_traces.tdms"
        if os.path.exists(traces_filename):
            # Determine chunk size of traces from the FL1index column
            sampleids = tdms_file.object("Cell Track", "FL1index").data
            traces_file = TdmsFile(traces_filename)
            for group, ch in dfn.tr_data:
                try:
                    trdat = traces_file.object(group, ch).data
                except KeyError:
                    pass
                else:
                    if trdat is not None:
                        # Only add trace if there is actual data.
                        # Split only needs the the position of the sections,
                        # so we remove the first (0) index.
                        self.traces[ch] = np.split(trdat, sampleids[1:])


    def ApplyFilter(self, force=[]):
        """ Computes the filters for the data set
        
        Uses `self._old_filters` to determine new filters.

        Parameters
        ----------
        force : list()
            A list of parameters that must be refiltered.


        Notes
        -----
        Updates `self.Configuration["Filtering"].
        
        The data is filtered according to filterable attributes in
        the global variable `dfn.cfgmap`.
        """

        # These lists may help us become very fast in the future
        newkeys = list()
        oldvals = list()
        newvals = list()
        
        if not "Filtering" in self.Configuration:
            self.Configuration["Filtering"] = dict()

        ## Determine which data was updated
        FIL = self.Configuration["Filtering"]
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
        attr2update = list()
        for k in newkeys:
            # k[:-4] because we want to crop " Min" and " Max"
            # "Polygon Filters" is not processed here.
            if k[:-4] in dfn.uid:
                attr2update.append(dfn.cfgmaprev[k[:-4]])


        if "deform" in attr2update:
            attr2update.append("circ")
        elif "circ" in attr2update:
            attr2update.append("deform")

        for f in force:
            # Check if a correct variable is forced
            if f in list(dfn.cfgmaprev.keys()):
                attr2update.append(dfn.cfgmaprev[f])
            else:
                warnings.warn(
                    "Unknown variable not force-filtered: {}".format(f))
        
        attr2update = np.unique(attr2update)

        for attr in attr2update:
            data = getattr(self, attr)
            if isinstance(data, np.ndarray):
                fstart = dfn.cfgmap[attr]+" Min"
                fend = dfn.cfgmap[attr]+" Max"
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
        pf_id = "Polygon Filters"
        if (
            (pf_id in FIL and not pf_id in OLD) or
            (pf_id in FIL and pf_id in OLD and
             FIL[pf_id] != OLD[pf_id])):
            self._filter_polygon[:] = True
            # perform polygon filtering
            for p in PolygonFilter.instances:
                if p.unique_id in FIL["Polygon Filters"]:
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
        for attr in dir(self):
            if attr.startswith("_filter_"):
                self._filter[:] *= getattr(self, attr)

        # Filter with configuration keyword argument "Limit Events"
        if FIL["Limit Events"] > 0:
            limit = FIL["Limit Events"]
            incl = self._filter.copy()
            numevents = np.sum(incl)
            if limit <= numevents:
                # Perform equally distributed removal of events
                # We have too many events
                remove = numevents - limit
                while remove > 10:
                    there = np.where(incl)[0]
                    # first remove evenly distributed events
                    dist = int(np.ceil(there.shape[0]/remove))
                    incl[there[::dist]] = 0
                    numevents = np.sum(incl)
                    remove = numevents - limit
                there = np.where(incl)[0]
                incl[there[:remove]] = 0
                self._filter_limit = incl
                print("'Limit Events' set to {}/{}".format(np.sum(incl), incl.shape[0]))
            elif limit == numevents:
                # everything is ok
                self._filter_limit = np.ones_like(self._filter)
                print("'Limit Events' is size of filtered data.")
            elif limit <= self._filter.shape[0]:
                self._filter_limit = np.ones_like(self._filter)
                warnings.warn("{}: 'Limit Events' must not ".format(self.name)+
                              "be larger than length of filtered data set! "+
                              "Resetting 'Limit Events'!")
                FIL["Limit Events"] = 0
            else:
                self._filter_limit = np.ones_like(self._filter)
                warnings.warn("{}: 'Limit Events' must not ".format(self.name)+
                              "be larger than length of data set! "+
                              "Resetting 'Limit Events'!")
                FIL["Limit Events"] = 0
        else:
            # revert everything back to how it was
            self._filter_limit = np.ones_like(self._filter)
        
        # Update filter again
        try:
            self._filter *= self._filter_limit
        except:
            import IPython
            IPython.embed() 
        
        # Actual filtering is then done during plotting            
        self._old_filters = copy.deepcopy(self.Configuration["Filtering"])


    def ExportTSV(self, path, columns, filtered=True, override=False):
        """ Export the data of the current instance to a .tsv file
        
        Parameters
        ----------
        path : str
            Path to a .tsv file. The ending .tsv is added automatically.
        columns : list of str
            The columns in the resulting .tsv file. These are strings
            that are defined in `dclab.definitions.uid`, e.g.
            "Area", "Defo", "Frame", "FL-1max", "Area Ratio".
        filtered : bool
            If set to ``True``, only the filtered data (index in self._filter)
            are used.
        override : bool
            If set to ``True``, an existing file ``path`` will be overridden.
            If set to ``False``, an ``OSError`` will be raised.
        """
        # Make sure that path ends with .tsv
        if not path.endswith(".tsv"):
            path += ".tsv"
        # Check if file already exist
        if not override and os.path.exists(path):
            raise OSError("File already exists: {}\n".format(
                                    path.encode("ascii", "ignore"))+
                          "Please use the `override=True` option.")
        # Check that columns are in dfn.uid
        for c in columns:
            assert c in dfn.uid, "Unknown column name {}".format(c)
        
        # Open file
        with codecs.open(path, "w", encoding="utf-8") as fd:
            # write header
            header1 = "\t".join([ c for c in columns ])
            fd.write("# "+header1+"\n")
            header2 = "\t".join([ dfn.axlabels[c] for c in columns ])
            fd.write("# "+header2+"\n")

        with open(path, "ab") as fd:
            # write data
            if filtered:
                data = [ getattr(self, dfn.cfgmaprev[c])[self._filter] for c in columns ]
            else:
                data = [ getattr(self, dfn.cfgmaprev[c]) for c in columns ]
            
            np.savetxt(fd,
                       np.array(data).transpose(),
                       fmt=str("%.10e"),
                       delimiter="\t")


    def GetDownSampledScatter(self, c=None, axsize=(300,300),
                              markersize=1,
                              downsample_events=None):
        """ Filters a set of data from overlayed events for plottinǵ
        
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
            self.Configuration["Plotting"]["Downsample Events"]
            will be used.
        
        Returns
        -------
        xnew, xnew : filtered x and y
        """
        plotfilters = self.Configuration["Plotting"]
        if downsample_events is None:
            downsample_events = plotfilters["Downsample Events"]

        if downsample_events < 1:
            downsample_events = 0

        xax, yax = self.GetPlotAxes()

        # identifier for this setup
        identifier = str(axsize)+str(markersize)+str(c)
        # Get axes
        if self.Configuration["Filtering"]["Enable Filters"]:
            x = getattr(self, dfn.cfgmaprev[xax])[self._filter]
            y = getattr(self, dfn.cfgmaprev[yax])[self._filter]
            identifier += str(self.Configuration["Filtering"])
        else:
            # filtering disabled
            x = getattr(self, dfn.cfgmaprev[xax])
            y = getattr(self, dfn.cfgmaprev[yax])

        identifier += str(downsample_events)
            
        hasher = hashlib.sha256()
        hasher.update(str(x) + str(y))
        identifier += hasher.hexdigest()

        if identifier in self._Downsampled_Scatter:
            return self._Downsampled_Scatter[identifier]

        if downsample_events > 0 and downsample_events > x.shape[0]:
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

        self._plot_filter = incl

        xincl = x[np.where(incl)]
        yincl = y[np.where(incl)]

        if c is None:
            result = xincl, yincl
        else: 
            dens = c[np.where(incl)]
            result = xincl, yincl, dens
        
        self._Downsampled_Scatter[identifier] = result

        return result


    def GetKDE_Contour(self, yax="Defo", xax="Area"):
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
        if xax is None or yax is None:
            xax, yax = self.GetPlotAxes()
            
        kde_type = self.Configuration["Plotting"]["KDE"].lower()
        # dummy area-circ
        deltaarea = self.Configuration["Plotting"]["Contour Accuracy "+xax]
        deltacirc = self.Configuration["Plotting"]["Contour Accuracy "+yax]
        
        # setup
        if self.Configuration["Filtering"]["Enable Filters"]:
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
            bwx = self.Configuration["Plotting"]["KDE Multivariate "+xax]
            bwy = self.Configuration["Plotting"]["KDE Multivariate "+yax]
            kde_kwargs["bw"] = [bwx, bwy]
        
        kde_fct = getattr(kde_methods, "kde_"+kde_type)

        density = kde_fct(**kde_kwargs)

        print("KDE contour {} time: ".format(kde_type), time.time()-a)

        return xmesh, ymesh, density


    def GetKDE_Scatter(self, yax="Defo", xax="Area", positions=None):
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
        if self.Configuration["Filtering"]["Enable Filters"]:
            x = getattr(self, dfn.cfgmaprev[xax])[self._filter]
            y = getattr(self, dfn.cfgmaprev[yax])[self._filter]
        else:
            x = getattr(self, dfn.cfgmaprev[xax])
            y = getattr(self, dfn.cfgmaprev[yax])

        kde_type = self.Configuration["Plotting"]["KDE"].lower()
        
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
            bwx = self.Configuration["Plotting"]["KDE Multivariate "+xax]
            bwy = self.Configuration["Plotting"]["KDE Multivariate "+yax]
            kde_kwargs["bw"] = [bwx, bwy]
        
        kde_fct = getattr(kde_methods, "kde_"+kde_type)

        density = kde_fct(**kde_kwargs)
        
        print("KDE scatter {} time: ".format(kde_type), time.time()-a)
        return density


    def GetPlotAxes(self):
        #return 
        p = self.Configuration["Plotting"]
        return [p["Axis X"], p["Axis Y"]]


    def PolygonFilterAppend(self, filt):
        """ Associates a Polygon Filter to the RTDC_DataSet
        
        filt can either be an integer or an instance of PolygonFilter
        """
        if isinstance(filt, PolygonFilter):
            uid=filt.unique_id
        elif isinstance(filt, (int, float)):
            uid=int(filt)
        else:
            raise ValueError(
                  "filt must be a number or instance of PolygonFilter.")
        # append item
        self.Configuration["Filtering"]["Polygon Filters"].append(uid)


    def PolygonFilterRemove(self, filt):
        """ Opposite of PolygonFilterAppend """
        if isinstance(filt, PolygonFilter):
            uid = filt.unique_id
        elif isinstance(filt, (int, float)):
            uid = int(filt)
        else:
            raise ValueError(
                  "filt must be a number or instance of PolygonFilter.")
        # remove item
        self.Configuration["Filtering"]["Polygon Filters"].remove(uid)


    def SetConfiguration(self):
        """ Import configuration of measurement
        
        Requires the files "MX_camera.ini" and "MX_para.ini" to be
        present in `self.fdir`. The string "MX_" is at the beginning of
        `self.name` (measurement identifier).
        
        This function is called during `__init__` and it is not
        necessary to run it twice.
        """
        self.Configuration = dict()
        self.UpdateConfiguration(dfn.cfg)
        mx = self.name.split("_")[0]
        camcfg = dfn.LoadConfiguration(os.path.join(self.fdir, mx+"_camera.ini"))
        self.UpdateConfiguration(camcfg)
        parcfg = dfn.LoadConfiguration(os.path.join(self.fdir, mx+"_para.ini"))
        self.UpdateConfiguration(parcfg)


    def UpdateConfiguration(self, newcfg):
        """ Update current configuration `self.Configuration`.
        
        Parameters
        ----------
        newcfg : dict
            Dictionary to update `self.Configuration`
        
        
        Returns
        -------
        None (dictionary is updated in place).
        
        
        Notes
        -----
        It is not required to update the entire configuration. Small
        changes can be made.
        """
        force = []
        # look for pixel size update first
        if ("Image" in newcfg and
            "Pix Size" in newcfg["Image"]):
            PIX = newcfg["Image"]["Pix Size"]
            self.area_um[:] = self.area * PIX**2
            force.append("Area")
        # look for frame rate update
        elif ("Framerate" in newcfg and
            "Frame Rate" in newcfg["Framerate"]):
            FR = newcfg["Framerate"]["Frame Rate"]
            # FR is in Hz
            self.time[:] = (self.frame - self.frame[0]) / FR
            force.append("Time")

        UpdateConfiguration(self.Configuration, newcfg)

        if "Filtering" in newcfg:
            # Only writing the new Mins and Maxs is not enough
            # We need to also set the _filter_* attributes.
            self.ApplyFilter(force=force)
        
        # Reset additional information
        self.Configuration["General"]["Cell Number"] =self.time.shape[0]


    def __eq__(self, mm):
        if self.file_hashes == mm.file_hashes:
            return True
        else:
            return False


def _hashfile(fname, blocksize=65536):
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
        warnings.warn("Non-standard directory naming scheme: {}".format(path))
        project = trail1
    return project


def UpdateConfiguration(oldcfg, newcfg):
    """ Update a configuration in librtdc format.
    
        
    Returns
    -------
    The new configuration, but it is also updated in-place.
    
    
    Notes
    -----
    Also converts from circularity to deformation in `newcfg`.
    """
    ## Defo to Circ conversion
    # new
    cmin = None
    cmax = None
    dmin = None
    dmax = None
    if "Filtering" in newcfg:
        if "Defo Max" in newcfg["Filtering"]:
            dmax = newcfg["Filtering"]["Defo Max"]
        if "Defo Min" in newcfg["Filtering"]:
            dmin = newcfg["Filtering"]["Defo Min"]
        if "Circ Max" in newcfg["Filtering"]:
            cmax = newcfg["Filtering"]["Circ Max"]
        if "Circ Min" in newcfg["Filtering"]:
            cmin = newcfg["Filtering"]["Circ Min"]
    # old
    cmino = None
    cmaxo = None
    dmino = None
    dmaxo = None
    if "Filtering" in oldcfg:
        if "Defo Max" in oldcfg["Filtering"]:
            dmaxo = oldcfg["Filtering"]["Defo Max"]
        if "Defo Min" in oldcfg["Filtering"]:
            dmino = oldcfg["Filtering"]["Defo Min"]
        if "Circ Max" in oldcfg["Filtering"]:
            cmaxo = oldcfg["Filtering"]["Circ Max"]
        if "Circ Min" in oldcfg["Filtering"]:
            cmino = oldcfg["Filtering"]["Circ Min"]
    # translation to new
    if cmin != cmino and cmin is not None:
        newcfg["Filtering"]["Defo Max"] = 1 - cmin
    if cmax != cmaxo and cmax is not None:
        newcfg["Filtering"]["Defo Min"] = 1 - cmax
    if dmin != dmino and dmin is not None:
        newcfg["Filtering"]["Circ Max"] = 1 - dmin
    if dmax != dmaxo and dmax is not None:
        newcfg["Filtering"]["Circ Min"] = 1 - dmax

    ## Contour
    if ("Plotting" in newcfg and
        "Contour Accuracy Circ" in newcfg["Plotting"] and
        not "Contour Accuracy Defo" in newcfg["Plotting"]):
        # If not contour accuracy for Defo is given, use that from Circ.
        newcfg["Plotting"]["Contour Accuracy Defo"] = newcfg["Plotting"]["Contour Accuracy Circ"]

    for key in list(newcfg.keys()):
        if not key in oldcfg:
            oldcfg[key] = dict()
        for skey in list(newcfg[key].keys()):
            oldcfg[key][skey] = newcfg[key][skey]

    ## Check missing values and set them to zero
    for item in dfn.uid:
        if not "Contour Accuracy "+item in oldcfg["Plotting"]:
            oldcfg["Plotting"]["Contour Accuracy "+item] = 1
        appends = [" Min", " Max"]
        for a in appends:
            if not item+a in oldcfg["Plotting"]:
                oldcfg["Plotting"][item+a] = 0
            if not item+a in oldcfg["Filtering"]:
                    oldcfg["Filtering"][item+a] = 0

    return oldcfg


def SaveConfiguration(cfgfilename, cfg):
    """ Save configuration to text file


    Parameters
    ----------
    cfgfilename : absolute path
        Filename of the configuration
    cfg : dict
        Dictionary containing configuration.

    """
    out = []
    keys = list(cfg.keys())
    keys.sort()
    for key in keys:
        out.append("[{}]".format(key))
        section = cfg[key]
        ikeys = list(section.keys())
        ikeys.sort()
        for ikey in ikeys:
            var, val = dfn.MapParameterType2Str(ikey, section[ikey])
            out.append("{} = {}".format(var,val))
        out.append("")
    
    f = codecs.open(cfgfilename, "wb", "utf-8")
    for i in range(len(out)):
        out[i] = out[i]+"\r\n"
    f.writelines(out)
    f.close()


