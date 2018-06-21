#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Class for efficiently handling image/video data
"""
from __future__ import division, print_function, unicode_literals

import pathlib
import sys
import warnings

import numpy as np
import imageio


ISWIN = sys.platform.startswith("win")


class ImageColumn(object):
    def __init__(self, rtdc_dataset):
        """A wrapper for ImageMap that takes into account event offsets

        Event offsets appear when the first event that is recorded in the
        tdms files does not have a corresponding cell image in the video
        file.
        """
        fname = self.find_video_file(rtdc_dataset)
        self.identifier = fname
        if fname is not None:
            self._image_data = ImageMap(fname)
        else:
            self._image_data = []
        conf = rtdc_dataset.config
        self.event_offset = int(conf["fmt_tdms"]["video frame offset"])
        self.video_file = fname

    def __getitem__(self, idx):
        idnew = int(idx-self.event_offset)
        if idnew < 0:
            # No data - show a dummy image instead
            cdata = self.dummy
        else:
            try:
                cdata = self._image_data[idnew]
            except imageio.plugins.ffmpeg.CannotReadFrameError:
                # The avi is corrupt. Return a dummy image.
                msg = "Frame {} in {} is corrupt!".format(idnew,
                                                          self.identifier)
                warnings.warn(msg)
                cdata = self.dummy
        return cdata

    def __len__(self):
        length = len(self._image_data)
        if length:
            length = length+self.event_offset
        return length

    @property
    def dummy(self):
        """Returns a dummy image"""
        cdata = np.zeros_like(self._image_data[0])
        return cdata

    @staticmethod
    def find_video_file(rtdc_dataset):
        """Tries to find a video file that belongs to an RTDC dataset

        Returns None if no video file is found.
        """
        video = None
        if rtdc_dataset._fdir.exists():
            # Cell images (video)
            videos = [v.name for v in rtdc_dataset._fdir.rglob("*.avi")]
            # Filter videos according to measurement number
            meas_id = rtdc_dataset._mid
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
            return rtdc_dataset._fdir / video


class ImageMap(object):
    def __init__(self, fname):
        """Access a video file of an RT-DC dataset

        Initialize this class with a video file.
        """
        fname = pathlib.Path(fname)
        if not fname.exists():
            raise OSError("file does not exist: {}".format(fname))
        self.filename = fname
        self._length = None
        # video handle:
        self._cap = None

    def __del__(self):
        if self._cap is not None:
            if ISWIN:
                # This is a workaround for windows when pytest fails due
                # to "OSError: [WinError 6] The handle is invalid",
                # which is somehow related to the fact that "_proc.kill()"
                # must be called twice (in "close()" and in this case) in
                # order to terminate the process and due to the fact the
                # we are not using the with-statement in combination
                # with imageio.get_reader().
                self._cap._proc.kill()
            self._cap.close()

    def __getitem__(self, idx):
        """Returns the requested frame from the video in gray scale"""
        cap = self.video_handle
        cellimg = cap.get_data(idx)
        if np.all(cellimg == 0):
            cellimg = self._get_image_workaround_seek(idx)
        # Convert to grayscale
        if len(cellimg.shape) == 3:
            cellimg = np.array(cellimg[:, :, 0])
        return cellimg

    def __len__(self):
        """Returns the length of the video or `True` if the length cannot be
        determined.
        """
        if self._length is None:
            cap = self.video_handle
            length = len(cap)
            self._length = length
        return self._length

    def _get_image_workaround_seek(self, idx):
        """Same as __getitem__ but seek through the video beforehand

        This is a workaround for an all-zero image returned by `imageio`.
        """
        warnings.warn("imageio workaround used!")
        cap = self.video_handle
        mult = 50
        for ii in range(idx//mult):
            cap.get_data(ii*mult)
        final = cap.get_data(idx)
        return final

    @property
    def video_handle(self):
        if self._cap is None:
            self._cap = imageio.get_reader(self.filename)
        return self._cap
