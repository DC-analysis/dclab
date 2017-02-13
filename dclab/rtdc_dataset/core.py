#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
RTDC_DataSet classes and methods
"""
from __future__ import division, print_function, unicode_literals

import codecs
import copy
from distutils.version import LooseVersion
import fcswrite
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
from .event_contour import ContourColumn
from .event_image import ImageColumn
from .event_trace import TraceColumn


if sys.version_info[0] == 2:
    str_classes = (str, unicode)
else:
    str_classes = str


try:
    import cv2
except:
    warnings.warn("Could not import opencv, video-related functions will not work!")
else:
    # Constants in OpenCV moved from "cv2.cv" to "cv2"
    if LooseVersion(cv2.__version__) < LooseVersion("3.0.0"):
        cv_const = cv2.cv
        cv_version3 = False
    else:
        cv_const = cv2
        cv_version3 = True



class RTDC_DataSet(object):
    """ An RTDC measurement object.
    
    Parameters
    ----------
    tdms_path: str
        Path to a '.tdms' file. Only one of `tdms_path and `ddict` can
        be specified.
    ddict: dict
        Dictionary with keys from `dclab.definitions.uid` (e.g. "Area", "Defo")
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
            self.Configuration = config.load_default_config()

        # Initialize variables and generate hashes
        self.tdms_filename = tdms_path
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

        self._complete_configuration_defaults()


    def _complete_configuration_defaults(self):
        """Add missing values to self.Configuation
        
        This includes values for:
        - Plotting | Contour Accuracy
        - Plotting | KDE Multivariate
        """
        keys = []
        for prop in dfn.rdv:
            if not np.allclose(getattr(self, prop), 0):
                # There are values for this uid
                keys.append(prop)
        
        # Plotting defaults
        accl = lambda a: (np.nanmax(a)-np.nanmin(a))/10
        defs = [["Contour Accuracy {}", accl],
                ["KDE Multivariate {}", accl],
               ]

        pltng = self.Configuration["Plotting"]
        for k in keys:
            for d, l in defs:
                var = d.format(dfn.cfgmap[k])
                if not var in pltng:
                    pltng[var] = l(getattr(self, k))


    def _init_data_with_dict(self, ddict):
        for key in ddict:
            setattr(self, dfn.cfgmaprev[key], np.array(ddict[key]))
        fill0 = np.zeros(len(ddict[key]))
        for key in dfn.rdv:
            if not hasattr(self, key):
                setattr(self, key, fill0)

        # Set up filtering
        self._init_filters()
        self.SetConfiguration()


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
        cfg = copy.deepcopy(hparent.Configuration)
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
        self.Configuration = config.load_default_config()
        config.update_config_dict(self.Configuration, cfg)
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
        Updates `self.Configuration["Filtering"].
        
        The data is filtered according to filterable attributes in
        the global variable `dfn.cfgmap`.
        """

        # These lists may help us become very fast in the future
        newkeys = []
        oldvals = []
        newvals = []
        
        if not "Filtering" in self.Configuration:
            self.Configuration["Filtering"] = {"Enable Filters":False}

        FIL = self.Configuration["Filtering"]

        # Check if we are a hierarchy child and if yes, update the
        # filtered events from the hierarchy parent.
        if ("Hierarchy Parent" in FIL and
            FIL["Hierarchy Parent"] != "None"):
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
        if FIL["Enable Filters"]:
            for attr in dir(self):
                if attr.startswith("_filter_"):
                    self._filter[:] *= getattr(self, attr)
    
            # Filter with configuration keyword argument "Limit Events"
            if FIL["Limit Events"] > 0:
                limit = FIL["Limit Events"]
                incl = self._filter.copy()
                numevents = np.sum(incl)
                if limit < numevents:
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
            self._filter *= self._filter_limit

        # Actual filtering is then done during plotting            
        self._old_filters = copy.deepcopy(self.Configuration["Filtering"])


    def ExportAVI(self, path, override=False):
        """Exports filtered event images to an avi file

        Parameters
        ----------
        path : str
            Path to a .tsv file. The ending .tsv is added automatically.
        filtered : bool
            If set to ``True``, only the filtered data (index in self._filter)
            are used.
        override : bool
            If set to ``True``, an existing file ``path`` will be overridden.
            If set to ``False``, an ``OSError`` will be raised.
        
        Notes
        -----
        Raises OSError if current data set does not contain image data
        """
        # TODO:
        # - Write tests for this method to keep dclab coverage close to 100%
        if len(self.image):
            # write the (filtered) images to an avi file
            # check for offset defined in para    
            video_file = self.image.video_file
            vReader = cv2.VideoCapture(video_file)
            if cv_version3:
                totframes = vReader.get(cv_const.CAP_PROP_FRAME_COUNT)
            else:
                totframes = vReader.get(cv_const.CV_CAP_PROP_FRAME_COUNT)
            # determine size of video
            f, i = vReader.read()
            print("video_file: ", video_file)
            print("Open: ", vReader.isOpened())
            print(vReader)
            #print("reading frame", f, i)
            if (f==False):
                print("Could not read AVI, abort.")
                return -1
            videoSize = (i.shape[1], i.shape[0])
            videoShape= i.shape
            # determine video file offset. Some RTDC setups
            # do not record the first image of a video.
            frames_skipped = self.image.event_offset
            # filename for avi output
            # Make sure that path ends with .tsv
            if not path.endswith(".avi"):
                path += ".avi"
            # Open destination video
            # use i420 code, as it is working on MacOS
            # fourcc = cv2.VideoWriter_fourcc('I','4','2','0')
            # error when running on mac... so give fourcc manually as number
            fourcc = 808596553
            if vReader.isOpened():
                vWriter = cv2.VideoWriter(path, fourcc, 25, videoSize, isColor=True)
            if vWriter.isOpened():
                # print(self._filter)
                # write the filtered frames to avi file
                for evId in np.arange(len(self._filter)):
                    # skip frames that were filtered out
                    if self._filter[evId] == False:
                        continue
                    # look for this frame in source video
                    fId = evId - frames_skipped
                    if fId < 0:
                        # get placeholder
                        print("fId < 0: get placeholder image")
                        i = np.zeros(videoShape, dtype=np.uint8)
                    elif fId >= totframes:
                        print("fId > total frames")
                        continue
                    else:
                        # get this frame
                        if cv_version3:
                            vReader.set(cv_const.CAP_PROP_POS_FRAMES, fId)
                        else:
                            vReader.set(cv_const.CV_CAP_PROP_POS_FRAMES, fId)
                        flag, i = vReader.read()
                        if not flag:
                            i = np.zeros(videoShape, dtype=np.uint8)
                            print("Could not read event/frame", evId, "/", fId)
                    # print("Write image ", evId,"/", totframes)
                    # for monochrome
                    # vWriter.write(i[:,:,0])
                    # for color images
                    vWriter.write(i)
                # and close it
                vWriter.release()
        else:
            msg="No video data to export from dataset {} !".format(self.title)
            raise OSError(msg)


    def ExportFCS(self, path, columns, filtered=True, override=False):
        """ Export the data of an RTDC_DataSet to an .fcs file
        
        Parameters
        ----------
        mm: instance of dclab.RTDC_DataSet
            The data set that will be exported.
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
        # TODO:
        # - Write tests for this method to keep dclab coverage close to 100%
        
        # Make sure that path ends with .fcs
        if not path.endswith(".fcs"):
            path += ".fcs"
        # Check if file already exist
        if not override and os.path.exists(path):
            raise OSError("File already exists: {}\n".format(
                                    path.encode("ascii", "ignore"))+
                          "Please use the `override=True` option.")
        # Check that columns are in dfn.uid
        for c in columns:
            assert c in dfn.uid, "Unknown column name {}".format(c)
        
        # Collect the header
        chn_names = [ dfn.axlabels[c] for c in columns ]
    
        # Collect the data
        if filtered:
            data = [ getattr(self, dfn.cfgmaprev[c])[self._filter] for c in columns ]
        else:
            data = [ getattr(self, dfn.cfgmaprev[c]) for c in columns ]
        
        data = np.array(data).transpose()
        fcswrite.write_fcs(filename=path,
                           chn_names=chn_names,
                           data=data)


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
            self.Configuration["Plotting"]["Downsample Events"]
            will be used.
        
        Returns
        -------
        xnew, xnew : filtered x and y
        """
        self.ApplyFilter()
        plotfilters = self.Configuration["Plotting"]
        if downsample_events is None:
            downsample_events = plotfilters["Downsample Events"]

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
        if self.Configuration["Filtering"]["Enable Filters"]:
            x = getattr(self, dfn.cfgmaprev[xax])[self._filter]
            y = getattr(self, dfn.cfgmaprev[yax])[self._filter]
            hasher.update(obj2str(self.Configuration["Filtering"]))
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
        
        if len(x) != 0:
            density = kde_fct(**kde_kwargs)
        else:
            density = []
        
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
        msg = "`filt` must be a number or instance of PolygonFilter."
        assert isinstance(filt, (PolygonFilter, int, float)), msg
        
        if isinstance(filt, PolygonFilter):
            uid=filt.unique_id
        else:
            uid=int(filt)
        # append item
        self.Configuration["Filtering"]["Polygon Filters"].append(uid)


    def PolygonFilterRemove(self, filt):
        """ Opposite of PolygonFilterAppend """
        msg = "`filt` must be a number or instance of PolygonFilter."
        assert isinstance(filt, (PolygonFilter, int, float)), msg
        
        if isinstance(filt, PolygonFilter):
            uid = filt.unique_id
        else:
            uid = int(filt)
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
        self.UpdateConfiguration(config.cfg)
        for name, _hash in self.file_hashes:
            if name.endswith(".ini") and os.path.exists(name):
                newdict = config.load_config_file(name)
                self.UpdateConfiguration(newdict)


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

        config.update_config_dict(self.Configuration, newcfg)

        if "Filtering" in newcfg:
            # Only writing the new Mins and Maxs is not enough
            # We need to also set the _filter_* attributes.
            self.ApplyFilter(force=force)
        
        # Reset additional information
        self.Configuration["General"]["Cell Number"] = self.time.shape[0]



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
