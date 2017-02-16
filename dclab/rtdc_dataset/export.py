#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Export RTDC_DataSet data
"""
from __future__ import division, print_function, unicode_literals

import codecs
from distutils.version import LooseVersion
import fcswrite
import numpy as np
import os
import platform
import warnings

from .. import definitions as dfn


try:
    import cv2
except ImportError:
    warnings.warn("OpenCV not available!")
else:
    # Constants in OpenCV moved from "cv2.cv" to "cv2"
    if LooseVersion(cv2.__version__) < LooseVersion("3.0.0"):
        CV_CAP_PROP_POS_FRAMES = cv2.cv.CV_CAP_PROP_POS_FRAMES
        CV_CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
        CV_FOURCC = cv2.cv.CV_FOURCC
    else:
        CV_CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
        CV_CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
        CV_FOURCC = cv2.VideoWriter_fourcc


class Export(object):
    def __init__(self, rtdc_ds):
        """Export functionalities for RTDC_DataSet"""
        self.rtdc_ds = rtdc_ds


    def avi(self, path, override=False):
        """Exports filtered event images to an avi file

        Parameters
        ----------
        path : str
            Path to a .tsv file. The ending .tsv is added automatically.
        filtered : bool
            If set to ``True``, only the filtered data (index in ds._filter)
            are used.
        override : bool
            If set to ``True``, an existing file ``path`` will be overridden.
            If set to ``False``, an ``OSError`` will be raised.
        
        Notes
        -----
        Raises OSError if current data set does not contain image data
        """
        # TODO:
        # - Write tests for this method
        # - Make video export work with mac, windows, and linux

        ds = self.rtdc_ds
        # Make sure that path ends with .avi
        if not path.endswith(".avi"):
            path += ".avi"
        # Check if file already exist
        if not override and os.path.exists(path):
            raise OSError("File already exists: {}\n".format(
                                    path.encode("ascii", "ignore"))+
                          "Please use the `override=True` option.")
        # Start exporting
        if len(ds.image):
            v_size = (ds.image[0].shape[1], ds.image[0].shape[0])
            # FourCC code will not work for all systems
            if platform.system() == "Darwin":
                fourcc = b"I420"
                # What about "AVC1"?
            elif platform.system() == "Windows":
                fourcc = b"DIB "
                #HFYU?
            else:
                # probably linux
                fourcc = b"HFYU"
            # Open video for writing
            vout = cv2.VideoWriter(path,
                                   fourcc=CV_FOURCC(*fourcc),
                                   fps=25,
                                   frameSize=v_size,
                                   isColor=True)
            if vout.isOpened():
                # write the filtered frames to avi file
                for evid in np.arange(len(ds._filter)):
                    # skip frames that were filtered out
                    if not ds._filter[evid]:
                        continue
                    try:
                        image = ds.image[evid]
                    except:
                        warnings.warn("Could not read image {}!".format(evid))
                        continue
                    vout.write(image)
                vout.release()
            else:
                raise OSError("Could not write Video.")
        else:
            msg="No image data to export: dataset {} !".format(ds.title)
            raise OSError(msg)


    def fcs(self, path, columns, filtered=True, override=False):
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
            If set to ``True``, only the filtered data (index in ds._filter)
            are used.
        override : bool
            If set to ``True``, an existing file ``path`` will be overridden.
            If set to ``False``, an ``OSError`` will be raised.
        """
        columns = [ c.lower() for c in columns ]
        ds = self.rtdc_ds
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
            data = [ getattr(ds, dfn.cfgmaprev[c])[ds._filter] for c in columns ]
        else:
            data = [ getattr(ds, dfn.cfgmaprev[c]) for c in columns ]
        
        data = np.array(data).transpose()
        fcswrite.write_fcs(filename=path,
                           chn_names=chn_names,
                           data=data)


    def tsv(self, path, columns, filtered=True, override=False):
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
            If set to ``True``, only the filtered data (index in ds._filter)
            are used.
        override : bool
            If set to ``True``, an existing file ``path`` will be overridden.
            If set to ``False``, an ``OSError`` will be raised.
        """
        columns = [ c.lower() for c in columns ]
        ds = self.rtdc_ds
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
                data = [ getattr(ds, dfn.cfgmaprev[c])[ds._filter] for c in columns ]
            else:
                data = [ getattr(ds, dfn.cfgmaprev[c]) for c in columns ]
            
            np.savetxt(fd,
                       np.array(data).transpose(),
                       fmt=str("%.10e"),
                       delimiter="\t")
