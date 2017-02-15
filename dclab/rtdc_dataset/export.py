#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Export RTDC_DataSet data
"""
from __future__ import division, print_function, unicode_literals

import codecs
from distutils.version import LooseVersion
import os
import numpy as np
import fcswrite
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
    else:
        CV_CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
        CV_CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT



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
        ds = self.rtdc_ds
        # TODO:
        # - Write tests for this method to keep dclab coverage close to 100%
        if len(ds.image):
            # write the (filtered) images to an avi file
            # check for offset defined in para    
            video_file = ds.image.video_file
            vReader = cv2.VideoCapture(video_file)
            totframes = vReader.get(CV_CAP_PROP_FRAME_COUNT)
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
            frames_skipped = ds.image.event_offset
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
                # print(ds._filter)
                # write the filtered frames to avi file
                for evId in np.arange(len(ds._filter)):
                    # skip frames that were filtered out
                    if ds._filter[evId] == False:
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
                        vReader.set(CV_CAP_PROP_POS_FRAMES, fId)
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
            msg="No video data to export from dataset {} !".format(ds.title)
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
