#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Class for efficiently handling image/video data
"""
from __future__ import division, print_function, unicode_literals

import pathlib
import sys
import warnings

import imageio
import numpy as np

from .exc import InvalidVideoFileError


ISWIN = sys.platform.startswith("win")


class CorruptFrameWarning(UserWarning):
    pass


class InitialFrameMissingWarning(CorruptFrameWarning):
    pass


class SlowVideoWarning(UserWarning):
    pass


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
        self._shape = None

    def __getitem__(self, idx):
        idnew = int(idx-self.event_offset)
        if idnew < 0:
            # No data - show a dummy image instead
            warnings.warn("Frame {} in {} ".format(idnew, self.identifier)
                          + "is not defined; replacing with dummy image!",
                          InitialFrameMissingWarning)
            cdata = self.dummy
        else:
            if hasattr(imageio.plugins.ffmpeg, "CannotReadFrameError"):
                # imageio<2.5.0
                excs = IndexError, imageio.plugins.ffmpeg.CannotReadFrameError
            else:
                # imageio>=2.5.0
                excs = IndexError
            try:
                cdata = self._image_data[idnew]
            except excs:
                # The avi is corrupt. Return a dummy image.
                warnings.warn("Frame {} in {} ".format(idnew, self.identifier)
                              + "is corrupt; replacing with dummy image!",
                              CorruptFrameWarning)
                cdata = self.dummy
        return cdata

    def __len__(self):
        length = len(self._image_data)
        if length:
            length = length + self.event_offset
        return length

    @property
    def dummy(self):
        """Returns a dummy image"""
        cdata = np.zeros(self.shape, dtype=np.uint8)
        return cdata

    @property
    def shape(self):
        if self._shape is None:
            f0 = self._image_data[0].shape
            self._shape = (f0[0], f0[1])
        return self._shape

    @staticmethod
    def find_video_file(rtdc_dataset):
        """Tries to find a video file that belongs to an RTDC dataset

        Returns None if no video file is found.
        """
        video = None
        if rtdc_dataset._fdir.exists():
            # Cell images (video)
            videos = [v.name for v in rtdc_dataset._fdir.glob("*.avi")]
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
            vpath = rtdc_dataset._fdir / video
            if vpath.stat().st_size < 64:
                # the video file is empty
                raise InvalidVideoFileError("Bad video '{}'!".format(vpath))
            else:
                return vpath


class ImageMap(object):
    def __init__(self, fname):
        """Access a video file of an RT-DC dataset

        Initialize this class with a video file.
        """
        fname = pathlib.Path(fname)
        self._length = None
        # video handle:
        self._cap = None
        # filename
        if not fname.exists():
            raise OSError("file does not exist: {}".format(fname))
        self.filename = fname

    def __del__(self):
        if self._cap is not None:
            if ISWIN and self._cap._proc is not None:
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
        try:
            cellimg = cap.get_data(idx)
        except IndexError:
            self._get_image_workaround_seek(idx)
        else:
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
            if hasattr(cap, "count_frames"):  # imageio>=2.5.0
                length = cap.count_frames()
            else:  # imageio<2.5.0
                try:
                    length = len(cap)
                except TypeError:
                    # length is set to inf (it would generally still be
                    # possible to rescue a limted number of frames, but
                    # other parts of the dataset are probably also broken
                    # (aborted measurement), so for the sake of simplicity
                    # we stop here)
                    raise InvalidVideoFileError(
                        "Video file has unknown length '{}'!".format(
                            self.filename))
            self._length = length
        return self._length

    def _get_image_workaround_seek(self, idx):
        """Same as __getitem__ but seek through the video beforehand

        This is a workaround for an all-zero image returned by `imageio`.
        """
        cap = self.video_handle
        mult = 50
        # get the first frame to be on the safe side
        cap.get_data(101)  # frame above 100 (don't know why)
        cap.get_data(0)  # seek to zero
        for ii in range(idx//mult):
            cap.get_data(ii*mult)
        final = cap.get_data(idx)
        if not np.all(final == 0):
            # This means we succeeded
            warnings.warn("Seeking video file does not work, used workaround "
                          + "which is slow!", SlowVideoWarning)
        return final

    @property
    def video_handle(self):
        if self._cap is None:
            try:
                self._cap = imageio.get_reader(self.filename)
            except OSError:
                raise InvalidVideoFileError(
                    "Broken video file '{}'!".format(self.filename))
        return self._cap
