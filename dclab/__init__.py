#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This library contains classes and methods for the analysis
of real-time deformability cytometry (RT-DC) data sets.
"""
from __future__ import division, print_function

import codecs
import copy
import hashlib
from nptdms import TdmsFile  # @UnresolvedImport
import numpy as np
import os
from scipy.stats import norm, gaussian_kde
from statsmodels.nonparametric.kernel_density import KDEMultivariate  # @UnresolvedImport
import time
    
import warnings

# Definitions
from . import definitions as dfn  # @UnresolvedImport
from ._version import version as __version__

class Fake_RTDC_DataSet(object):
    """ Provides methods and attributes like RTDC_DataSet, but without
        data.
    
    Needs a `Configuration` (e.g. from an RTDC_DataSet).
    """
    def __init__(self, Configuration):
        for item in dfn.rdv:
            setattr(self, item, np.zeros(10))
        
        self.deform +=1
        self.area_um +=1
        self._filter =  np.ones(10, dtype=bool)
        self.Configuration = copy.deepcopy(Configuration)
        self.Configuration["Plotting"]["Contour Color"] = "white"
        self.name = ""
        self.tdms_filename = ""
        self.title = ""
        self.file_hashes = [["None", "None"]]
        self.identifier = "None"

    def GetDownSampledScatter(self, *args, **kwargs):
        return np.zeros(10), np.zeros(10)
        
    def GetKDE_Contour(self, yax="Defo", xax="Area"):
        return [[np.zeros(1)]*3]*3
    
    def GetKDE_Scatter(self, yax="Defo", xax="Area", positions=None):
        return np.zeros(10)

    def UpdateConfiguration(self, newcfg):
        UpdateConfiguration(self.Configuration, newcfg)

        
class PolygonFilter(object):
    """ An object for filtering RTDC data based on a polygonial area
    """
    # Stuff that is done upon creation (not instantiation) of this class
    instances = []
    _instance_counter = 0
        
    def __init__(self, axes=None, points=None, name=None,
                 filename=None, fileid=0, unique_id=None):
        """ Instantiates PolygonFilter
        
        Parameters
        ----------
        axes : tuple of str
            The axes on which the polygon is defined. The first axis is
            the x-axis. Example: ("Area", "Defo").
        points : array-like object of shape (N,2)
            The N coordinates (x,y) of the polygon. The exact order is
            important.
        name : str
            A name for the polygon (optional).
        filename : str (path)
            A path to a .poly file as create by this classes' `save`
            method. If filename is given, all other parameters are
            ignored.
        fileid : int
            Which filter to import from the file (starting at 0).
        unique_id : int
            An integer defining the unique id of the new instance.
        
        
        Notes
        -----
        The minimal arguments to this class are either `filename` OR
        (`axes`, `points`). If `filename` is set, all parameters are
        taken from the given .poly file.
        """
        # set our id and increment the _instance_counter
        if (unique_id is not None):
            if PolygonFilter.instace_exists(unique_id):
                raise ValueError("Instance with id {} already exists".
                                 format(unique_id))
            else:
                self.unique_id = unique_id
                PolygonFilter._instance_counter = max(
                           PolygonFilter._instance_counter, unique_id+1)
        else:
            self.unique_id = PolygonFilter._instance_counter
            PolygonFilter._instance_counter += 1

        # check if a filename was given
        if filename is not None:
            assert isinstance(fileid, int)
            assert os.path.exists(filename),\
                   "Error, no such file: {}".format(filename)
            self.fileid = fileid
            self._load(filename)

            # we are done here
        else:
            self.axes = axes
            self.points = np.array(points, dtype=float)
            self.name = name
        
        self._check_data()
        # if everything worked out, add to instances
        PolygonFilter.instances.append(self)
    
    
    def _check_data(self):
        """ Checks if the given data is valid.
        """
        assert self.axes is not None, "Error, `axes` parm not set."
        assert self.points is not None, "Error, `points` parm not set."
        self.points = np.array(self.points)
        assert self.points.shape[1] == 2, \
               "Error, data points must be have two coordinates."
        if self.name is None:
            self.name = "polygon filter {}".format(self.unique_id)
    
    @staticmethod
    def clear_all_filters():
        """ Removes all filters and resets instance counter.
        """
        PolygonFilter.instances = []
        PolygonFilter._instance_counter = 0
        
    
    def copy(self):
        """ Returns a copy of the current instance.
        """
        return PolygonFilter(axes=self.axes,
                             points=self.points,
                             name=self.name)
    
    
    def filter(self, datax, datay):
        """ Filters a set of datax and datay according to self.points.
        """
        f = np.ones(datax.shape, dtype=bool)
        for i, (x,y) in enumerate(zip(datax, datay)):
            f[i] = PolygonFilter.point_in_poly(x, y, self.points)
        return f
    
    
    @staticmethod
    def get_instance_from_id(unique_id):
        """ Returns the instance of the PolygonFilter with this
        unique_id.
        """
        for instance in PolygonFilter.instances:
            if instance.unique_id == unique_id:
                return instance
        # if this does not work:
        raise ValueError("PolygonFilter with unique_id {} not found.".
                         format(unique_id))
    
    @staticmethod
    def import_all(path):
        """ Imports all polygons from a .poly file.
        
        Returns a list of the imported polygon filters
        """
        plist = []
        fid = 0
        while True:
            try:
                p = PolygonFilter(filename=path, fileid=fid)
                plist.append(p)
                fid += 1
            except IndexError:
                break
        return plist
    
    @staticmethod
    def instace_exists(unique_id):
        """ Returns True if an instance with this id exists
        """
        try:
            PolygonFilter.get_instance_from_id(unique_id)
        except ValueError:
            return True
        else:
            return False
    
    def _load(self, filename):
        """ Imports all data from a text file.
        
        """
        fobj = open(filename, "r")
        data = fobj.readlines()
        fobj.close()

        # Get the strings that correspond to self.fileid
        bool_head = [ l.strip().startswith("[") for l in data ]
        
        int_head = np.squeeze(np.where(bool_head))
        int_head = np.atleast_1d(int_head)
        
        start = int_head[self.fileid]+1
        
        if len(int_head) > self.fileid+1:
            end = int_head[self.fileid+1]
        else:
            end = len(data)
        
        subdata = data[start:end]
        
        # separate all elements and strip them
        subdata = [ [ it.strip() for it in l.split("=") ] for l in subdata ]

        points = list()
        
        for var, val in subdata:
            if var.lower() == "x axis":
                xaxis = val
            elif var.lower() == "y axis":
                yaxis = val
            elif var.lower() == "name":
                self.name = val
            elif len(var) == 0:
                pass
            elif var.lower().startswith("point"):
                val = np.array(val.strip("[]").split(), dtype=float)
                points.append([int(var[5:]), val])
            else:
                raise NotImplementedError("Unknown variable: {} = {}".
                                          format(var, val))
        self.axes = (xaxis, yaxis)
        # sort points
        points.sort()
        # get only coordinates from points
        self.points = np.array([ p[1] for p in points ])
        
        # overwrite unique id
        self.unique_id = int(data[start-1].strip().strip("Polygon []"))

    @staticmethod
    def point_in_poly(x, y, poly):
        n = len(poly)
        inside = False

        p1x,p1y = poly[0]
        for i in range(n+1):
            p2x,p2y = poly[i % n]
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        if p1y != p2y:
                            xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x,p1y = p2x,p2y

        return inside

    @staticmethod
    def remove(unique_id):
        """ Removes a polygon filter with the unique_id from
            PolygonFilter.instances
        """
        for p in PolygonFilter.instances:
            if p.unique_id == unique_id:
                PolygonFilter.instances.remove(p)
        
    
    def save(self, polyfile, ret_fobj=False):
        """ Saves all data to a text file (appends data if file exists).
        
        Polyfile can be either a path to a file or a file object that
        was opened with the write "w" parameter. By using the file
        object, multiple instances of this class can write their data.
        
        If `ret_fobj` is `True`, then the file object will not be
        closed and returned. 
        """
        if polyfile.__class__ in [file]:
            fobj = polyfile
        elif polyfile.__class__ in [str, unicode]:
            fobj = open(polyfile, "a")
        else:
            raise ValueError("Argument {} must be a file object or "+\
                             "a path to a file.")
        
        # Who the hell would use more then 10 million polygons or
        # polygon points? -> 08d (easier if other people want to import)
        data2write = []
        data2write.append("[Polygon {:08d}]".format(self.unique_id))
        data2write.append("X Axis = {}".format(self.axes[0]))
        data2write.append("Y Axis = {}".format(self.axes[1]))
        data2write.append("Name = {}".format(self.name))
        for i, point in enumerate(self.points):
            data2write.append("point{:08d} = {:.15e} {:.15e}".format(i,
                                                            point[0],
                                                            point[1]))
        # Add new lines
        for i in range(len(data2write)):
            data2write[i] += "\n"

        # begin writing to fobj        
        fobj.writelines(data2write)
        
        if ret_fobj:
            return fobj
        else:
            fobj.close()
    
    @staticmethod
    def save_all(polyfile):
        """ Save all polygon filters
        """
        if len(PolygonFilter.instances) == 0:
            raise IndexError("There are not polygon filters to save.")
        for p in PolygonFilter.instances:
            # we return the ret_obj, so we don't need to open and
            # close the file multiple times.
            polyfile = p.save(polyfile, ret_fobj=True)
        polyfile.close()
        

class RTDC_DataSet(object):
    """ An RTDC measurement object.
    
    The object must be initiated with a '.tdms' filename.
    """
    def __init__(self, tdms_filename):
        """ Load tdms file and set all variables """
        # Kernel density estimator dictionaries
        self._KDE_Scatter = {}
        self._KDE_Contour = {}
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
        for i, group in enumerate(dfn.tfd):
            table = group[0]
            if not isinstance(group[1], list):
                group[1] = [group[1]]
            func = group[2]
            args = []
            try:
                for arg in group[1]:
                    data = tdms_file.object(table, arg).data
                    args.append(data)
            except KeyError:
                # set it to zero
                func = lambda x: x
                args = [np.zeros(datalen)]
            finally:
                setattr(self, dfn.rdv[i], func(*args))

        # Plotting filters, set by "GetDownSampledScatter".
        # This is a nested filter which is applied after self._filter
        self._plot_filter = np.ones_like(self.time, dtype=bool)

        # Set array filters:
        # This is the filter that will be used for plotting:
        self._filter = np.ones_like(self.time, dtype=bool)
        attrlist = dir(self)
        # Find attributes to be filtered
        # These are the filters from which self._filter is computed
        inifilter = np.ones(data.shape, dtype=bool)
        for attr in attrlist:
            # only allow filterable attributes from global dfn.cfgmap
            if not dfn.cfgmap.has_key(attr):
                continue
            data = getattr(self, attr)
            if isinstance(data, np.ndarray):
                # great, we are dealing with an array
                setattr(self, "_filter_"+attr, inifilter.copy())
        self._filter_polygon = inifilter.copy()

        self.SetConfiguration()

        # Get video file name
        videos = []
        for f in os.listdir(self.fdir):
            if f.endswith(".avi") and f.startswith(self.name[:2]):
                videos.append(f)
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


    def GetDownSampledScatter(self, c=None, axsize=(300,300),
                              markersize=1,
                              downsample_points=None):
        """ Filters a set of data from overlayed points
        
        Parameters
        ----------
        c : 1d array of same length as x and y
            Value (e.g. kernel density) for each point (x,y)
        axsize : size tuple
            Size of the axis.
        markersize : float
            Size of the marker (in dots), including edge.
        downsample_points : int or None
            Number of points to draw in the down-sampled plot.
            This number is either 
            - >=1: limit total number of points drawn
            - <1: only perform 1st downsampling step with grid
            If set to None, then
            self.Configuration["Plotting"]["Downsample Points"]
            will be used.
        
        Returns
        -------
        xnew, xnew : filtered x and y
        """
        plotfilters = self.Configuration["Plotting"]
        if downsample_points is None:
            downsample_points = plotfilters["Downsample Points"]

        if downsample_points < 1:
            downsample_points = 0

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

        identifier += str(downsample_points)
            
        hasher = hashlib.sha256()
        hasher.update(str(x) + str(y))
        identifier += hasher.hexdigest()

        if self._Downsampled_Scatter.has_key(identifier):
            return self._Downsampled_Scatter[identifier]

        if downsample_points > 0 and downsample_points > x.shape[0]:
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
        
        pointmask = mask.copy()
        
        for i in range(len(x)):
            xi = xpx[i]
            yi = ypx[i]
            ## first filter for exactly overlapping points
            if not pointmask[int(xi-1), int(yi-1)]:
                continue
            pointmask[int(xi-1), int(yi-1)] = False
            #boolvals = (xi-gridx)**2 + (yi-gridy)**2 < Rsq
            ## second filter for multiple overlay
            boolvals = (np.abs(xi-gridx) < D) * (np.abs(yi-gridy) < D)
            if np.sum(mask[boolvals]) != 0:
                mask *= np.logical_not(boolvals)
                #mask = np.logical_and(np.logical_not(boolvals), mask)
                #mask[boolvals] = 0
                incl[i] = True
        print("downsample time:", time.time()-a)


        # Perform upsampling: include points to match downsample_points
        if downsample_points > 0:
            numpoints = np.sum(incl)
            if downsample_points < numpoints:
                # Perform equally distributed removal of points
                # We have too many points
                remove = numpoints - downsample_points
                while remove > 10:
                    there = np.where(incl)[0]
                    # first remove evenly distributed points
                    dist = int(np.ceil(there.shape[0]/remove))
                    incl[there[::dist]] = 0
                    numpoints = np.sum(incl)
                    remove = numpoints - downsample_points
                there = np.where(incl)[0]
                incl[there[:remove]] = 0
            else:
                # Add equally distributed points
                # We have not enough points
                add = downsample_points - numpoints
                while add > 10:
                    away = np.where(~incl)[0]
                    # first remove evenly distributed points
                    dist = int(np.ceil(away.shape[0]/add))
                    incl[away[::dist]] = 1
                    numpoints = np.sum(incl)
                    add = downsample_points - numpoints
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

        # kernel density estimator
        # Ask Christoph H. about kernel density estimator, he has an other library
        # which allows for manual setting of the bandwidth parameter
        key = yax+"+"+xax+"_"+kde_type+str(deltaarea)+str(deltacirc)
        
        if kde_type == "multivariate":
            bwx = self.Configuration["Plotting"]["KDE Multivariate "+xax]
            bwy = self.Configuration["Plotting"]["KDE Multivariate "+yax]
            key += "_bw{}+{}_".format(bwx,bwy)

        # make sure the density is only used for the same set of
        # filters.
        if self.Configuration["Filtering"]["Enable Filters"]:
            key += str(self.Configuration["Filtering"]).strip("{}")

        if not self._KDE_Contour.has_key(key):
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
            Xmesh,Ymesh = np.meshgrid(xlin,ylin)
            X = Xmesh.ravel()
            Y = Ymesh.ravel()
            if kde_type == "gauss":
                estimator = gaussian_kde([x,y])
                Z = estimator.evaluate([X,Y]).reshape(len(ylin),len(xlin))
            elif kde_type == "multivariate":
                estimator_ly = KDEMultivariate(data=[x,y],var_type='cc',
                                               bw=[bwx, bwy])
                Z = estimator_ly.pdf([X,Y]).reshape(len(ylin),len(xlin))
            else:
                raise ValueError("Unknown KDE estimator {}".format(
                                                              kde_type))                
            self._KDE_Contour[key] = (Xmesh,Ymesh,Z)
        return self._KDE_Contour[key]



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
        
        TODO
        ----
        Do not use positions for the hasher. If the plot is filtered
        with marker size, we might end up computing the same KDE for
        the same points over and over again.
        """
        # Dictionary for KDE
        # kernel density estimator
        # Ask Christoph H. about kernel density estimator, he has an other library
        # which allows for manual setting of the bandwidth parameter
        
        kde_type = self.Configuration["Plotting"]["KDE"].lower()
        
        # make sure the density is used for only this set of variables
        key = yax+"+"+xax+"_"+kde_type
        if kde_type == "multivariate":
            bwx = self.Configuration["Plotting"]["KDE Multivariate "+xax]
            bwy = self.Configuration["Plotting"]["KDE Multivariate "+yax]
            key += "_bw{}+{}_".format(bwx,bwy)
        # make sure the density is only used for the same set of
        # filters.
        if self.Configuration["Filtering"]["Enable Filters"]:
            key += str(self.Configuration["Filtering"]).strip("{}")

        if positions is not None:
            # compute hash of positions
            hasher = hashlib.sha256()
            hasher.update(positions)
            key += hasher.hexdigest()
        
        if not self._KDE_Scatter.has_key(key):
            if self.Configuration["Filtering"]["Enable Filters"]:
                x = getattr(self, dfn.cfgmaprev[xax])[self._filter]
                y = getattr(self, dfn.cfgmaprev[yax])[self._filter]
            else:
                x = getattr(self, dfn.cfgmaprev[xax])
                y = getattr(self, dfn.cfgmaprev[yax])
            
            if kde_type == "gauss":
                input_positions = np.vstack([x.ravel(), y.ravel()])
                estimator = gaussian_kde(input_positions)
                if positions is None:
                    positions = input_positions
                a = time.time()
                density = estimator(positions)
                print("gaussian estimation scatter time: ", time.time()-a)
                
            elif kde_type == "multivariate":
                a = time.time()
                estimator_ly = KDEMultivariate(data=[x,y],var_type='cc',
                                               bw=[bwx, bwy])
                if positions is None:
                    positions = input_positions
                density = estimator_ly.pdf(positions)
                print("multivariate estimation scatter time: ", time.time()-a)
            else:
                raise ValueError("Unknown KDE estimator {}".format(
                                                              kde_type))
            self._KDE_Scatter[key] = density
        return self._KDE_Scatter[key]


    def PolygonFilterAppend(self, filt):
        """ Associates a Polygon Filter to the RTDC_DataSet
        
        filt can either be an integer or an instance of PolygonFilter
        """
        if isinstance(filt, PolygonFilter):
            self._polygon_filter_ids.append(filt.unique_id)
        elif isinstance(filt, (int, float)):
            self._polygon_filter_ids.append(int(filt))
        else:
            raise ValueError(
                  "filt must be a number or instance of PolygonFilter.")


    def PolygonFilterRemove(self, filt):
        """ Opposite of PolygonFilterAppend """
        if isinstance(filt, PolygonFilter):
            uid = filt.unique_id
        elif isinstance(filt, (int, float)):
            uid = int(filt)
        else:
            raise ValueError(
                  "filt must be a number or instance of PolygonFilter.")
        # remove from list
        self._polygon_filter_ids.remove(uid)
        

    def GetPlotAxes(self):
        #return 
        p = self.Configuration["Plotting"]
        return [p["Axis X"], p["Axis Y"]]


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


    def ApplyFilter(self, force=[]):
        """ Computes the filters for data plotting
        
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
        
        if not self.Configuration.has_key("Filtering"):
            self.Configuration["Filtering"] = dict()

        ## Determine which data was updated
        FIL = self.Configuration["Filtering"]
        OLD = self._old_filters
        
        for skey in list(FIL.keys()):
            if not OLD.has_key(skey):
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
                if (FIL.has_key(fstart) and FIL.has_key(fend) and
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
        if ((FIL.has_key(pf_id) and not OLD.has_key(pf_id)) or
            (FIL.has_key(pf_id) and OLD.has_key(pf_id) and
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

        # now update the entire object filter
        # get a list of all filters
        self._filter[:] = True
        for attr in dir(self):
            if attr.startswith("_filter_"):
                self._filter[:] *= getattr(self, attr)
        # Actual filtering is then done during plotting            
        
        self._old_filters = self.Configuration["Filtering"].copy()


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
        if (newcfg.has_key("Image") and
           newcfg["Image"].has_key("Pix Size")):
            PIX = newcfg["Image"]["Pix Size"]
            self.area_um[:] = self.area * PIX**2
            force.append("Area")
        # look for frame rate update
        elif (newcfg.has_key("Framerate") and
           newcfg["Framerate"].has_key("Frame Rate")):
            FR = newcfg["Framerate"]["Frame Rate"]
            # FR is in Hz
            self.time[:] = (self.frame - self.frame[0]) / FR
            force.append("Time")

        UpdateConfiguration(self.Configuration, newcfg)

        if newcfg.has_key("Filtering"):
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


def crop_linear_data(data, xmin, xmax, ymin, ymax):
    """ Crop plotting data.
    
    Crops plotting data of monotonous function and linearly interpolates
    values at end of interval.
    
    Paramters
    ---------
    data : ndarray of shape (N,2)
        The data to be filtered in x and y.
    xmin : float
        minimum value for data[:,0]
    xmax : float
        maximum value for data[:,0]
    ymin : float
        minimum value for data[:,1]
    ymax : float
        maximum value for data[:,1]    
    
    
    Returns
    -------
    ndarray of shape (M,2), M<=N
    
    Notes
    -----
    `data` must be monotonically increasing in x and y.
    
    """
    # TODO:
    # Detect re-entering of curves into plotting area
    x = data[:,0].copy()
    y = data[:,1].copy()
    
    # Filter xmin
    if np.sum(x<xmin) > 0:
        idxmin = np.sum(x<xmin)-1
        xnew = x[idxmin:].copy()
        ynew = y[idxmin:].copy()
        xnew[0] = xmin
        ynew[0] = np.interp(xmin, x, y)
        x = xnew
        y = ynew


    # Filter ymax
    if np.sum(y>ymax) > 0:
        idymax = len(y)-np.sum(y>ymax)+1
        xnew = x[:idymax].copy()
        ynew = y[:idymax].copy()
        ynew[-1] = ymax
        xnew[-1] = np.interp(ymax, y, x)
        x = xnew
        y = ynew
        

    # Filter xmax
    if np.sum(x>xmax) > 0:
        idxmax = len(y)-np.sum(x>xmax)+1
        xnew = x[:idxmax].copy()
        ynew = y[:idxmax].copy()
        xnew[-1] = xmax
        ynew[-1] = np.interp(xmax, x, y)
        x = xnew
        y = ynew
        
    # Filter ymin
    if np.sum(y<ymin) > 0:
        idymin = np.sum(y<ymin)-1
        xnew = x[idymin:].copy()
        ynew = y[idymin:].copy()
        ynew[0] = ymin
        xnew[0] = np.interp(ymin, y, x)
        x = xnew
        y = ynew
    
    newdata = np.zeros((len(x),2))
    newdata[:,0] = x
    newdata[:,1] = y

    return newdata

   
def GetTDMSFiles(directory):
    """ Recursively find projects based on '.tdms' file endings
    
    Searches the `directory` recursively for '.tdms' project files.
    Returns a list of files.
    
    If the callback function is defined, it will be called for each
    directory.
    """
    directory = os.path.realpath(directory)
    tdmslist = list()
    for root, _dirs, files in os.walk(directory):
        for f in files:
            # Philipp:
            # Exclude traces files of fRT-DC setup
            if (f.endswith(".tdms") and (not f.endswith("_traces.tdms"))):
                tdmslist.append(os.path.realpath(os.path.join(root,f)))
    tdmslist.sort()
    return tdmslist


def GetPolygonFilterNames():
    """ Returns the names of all polygon filters in the order of
        creation.
    """
    names = []
    for p in PolygonFilter.instances:
        names.append(p.name)
    return names


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
        warnings.warn("Unknown folder structure: {}".format(path))
        project = trail1
    return project


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
    if newcfg.has_key("Filtering"):
        if newcfg["Filtering"].has_key("Defo Max"):
            dmax = newcfg["Filtering"]["Defo Max"]
        if newcfg["Filtering"].has_key("Defo Min"):
            dmin = newcfg["Filtering"]["Defo Min"]
        if newcfg["Filtering"].has_key("Circ Max"):
            cmax = newcfg["Filtering"]["Circ Max"]
        if newcfg["Filtering"].has_key("Circ Min"):
            cmin = newcfg["Filtering"]["Circ Min"]
    # old
    cmino = None
    cmaxo = None
    dmino = None
    dmaxo = None
    if oldcfg.has_key("Filtering"):
        if oldcfg["Filtering"].has_key("Defo Max"):
            dmaxo = oldcfg["Filtering"]["Defo Max"]
        if oldcfg["Filtering"].has_key("Defo Min"):
            dmino = oldcfg["Filtering"]["Defo Min"]
        if oldcfg["Filtering"].has_key("Circ Max"):
            cmaxo = oldcfg["Filtering"]["Circ Max"]
        if oldcfg["Filtering"].has_key("Circ Min"):
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
    if (newcfg.has_key("Plotting") and
       newcfg["Plotting"].has_key("Contour Accuracy Circ")):
        newcfg["Plotting"]["Contour Accuracy Defo"] = newcfg["Plotting"]["Contour Accuracy Circ"]
    

    for key in list(newcfg.keys()):
        if not oldcfg.has_key(key):
            oldcfg[key] = dict()
        for skey in list(newcfg[key].keys()):
            oldcfg[key][skey] = newcfg[key][skey]

    ## Check missing values and set them to zero
    for item in dfn.uid:
        if not oldcfg["Plotting"].has_key("Contour Accuracy "+item):
            oldcfg["Plotting"]["Contour Accuracy "+item] = 1
        appends = [" Min", " Max"]
        for a in appends:
            if not oldcfg["Plotting"].has_key(item+a):
                oldcfg["Plotting"][item+a] = 0
            if not oldcfg["Filtering"].has_key(item+a):
                    oldcfg["Filtering"][item+a] = 0

    return oldcfg
    

def _get_data_path():
    return os.path.realpath(os.path.dirname(__file__))


def _hashfile(fname, blocksize=65536):
    afile = open(fname, 'rb')
    hasher = hashlib.sha256()
    buf = afile.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read(blocksize)
    afile.close()
    return hasher.hexdigest()



