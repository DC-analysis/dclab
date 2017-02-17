#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Class for efficiently handling image/video data
"""
from __future__ import division, print_function, unicode_literals

from distutils.version import LooseVersion
import numpy as np
import os
import warnings


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



class ImageColumn(object):
    def __init__(self, rtdc_dataset):
        """A wrapper for ImageMap that takes into account event offsets
        
        Event offsets appear when the first event that is recorded in the
        tdms files does not have a corresponding cell image in the video
        file.
        """
        fname = self.find_video_file(rtdc_dataset)
        if fname is not None:
            self._image_data = ImageMap(fname)
        else:
            self._image_data = []
        conf = rtdc_dataset.config
        self.event_offset = conf["general"]["video frame offset"]
        self.video_file = fname
        

    def __getitem__(self, idx):
        idnew = int(idx-self.event_offset)
        if idnew < 0:
            # No data - show a nice transition image instead
            cdata = self.dummy
        else:
            cdata = self._image_data[idnew]
        return cdata


    def __len__(self):
        length = len(self._image_data)
        if length:
            length = length+self.event_offset
        return length


    @property
    def dummy(self):
        """Returns a dummy image"""
        cdata = np.ones_like(self._image_data[0])
        cdata *= np.linspace(0,255,cdata.shape[0],
                             dtype=np.uint16).reshape(-1,1,1)
        return cdata


    @staticmethod
    def find_video_file(rtdc_dataset):
        """Tries to find a video file that belongs to an RTDC data set
        
        Returns None if no video file is found.
        """
        video = None
        fdir = rtdc_dataset.fdir
        if os.path.exists(fdir):
            # Cell images (video)
            videos = [v for v in os.listdir(fdir) if v.endswith(".avi")]
            # Filter videos according to measurement number
            meas_id = rtdc_dataset.name.split("_")[0]
            videos = [v for v in videos if v.split("_")[0] == meas_id] 
            videos.sort()
            if len(videos) != 0:
                # Defaults to first avi file
                video = videos[0]
                # g/q video file names. q comes first.
                for v in videos:
                    if v.endswith("imag.avi"):
                        video = v
                        break
                    # add this here, because fRT-DC measurements also contain
                    # videos ..._proc.avi
                    elif v.endswith("imaq.avi"):
                        video = v
                        break
        if video is None:
            return None
        else:
            return os.path.join(fdir, video)



class ImageMap(object):
    def __init__(self, fname):
        """Access a video file of an RT-DC data set
        
        Initialize this class with a video file.
        """
        assert os.path.exists(fname)
        self.filename = fname
        self._length = None


    def __getitem__(self, idx):
        """Returns the requested frame from the video in RGB"""
        cap = self._get_video_handler()
        cap.set(CV_CAP_PROP_POS_FRAMES, idx)
        flag, cellimg = cap.read()
        cap.release()
        if flag:
            if len(cellimg.shape) == 2:
                # convert grayscale to color
                cellimg = np.tile(cellimg, [3,1,1]).transpose(1,2,0)
        else:
            msg = "Could not read frame {} from {}".format(idx, self.filename)
            raise OSError(msg)
        
        return cellimg


    def __len__(self):
        """Returns the length of the video or `True` if the length cannot be
        determined.
        """
        if self._length is None:
            cap = self._get_video_handler()
            length = int(cap.get(CV_CAP_PROP_FRAME_COUNT))
            cap.release()
            self._length = length
        return self._length
    
    
    def _get_video_handler(self):
        old_dir = os.getcwd()
        fdir, vfile = os.path.split(self.filename)
        os.chdir(fdir)
        cap = cv2.VideoCapture(vfile)
        if not cap.isOpened():
            cap.open(vfile)
        os.chdir(old_dir)
        return cap

