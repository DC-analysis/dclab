#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Export RT-DC measurement data"""
from __future__ import division, print_function, unicode_literals

import io
import imageio
import fcswrite
import numpy as np
import os
import warnings

from .. import definitions as dfn


class Export(object):
    def __init__(self, rtdc_ds):
        """Export functionalities for RT-DC datasets"""
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
        # Make sure that path ends with .avi
        if not path.endswith(".avi"):
            path += ".avi"
        # Check if file already exist
        if not override and os.path.exists(path):
            raise OSError("File already exists: {}\n".format(
                                    path.encode("ascii", "ignore"))+
                          "Please use the `override=True` option.")
        # Start exporting
        if "image" in ds:
            # Open video for writing
            vout = imageio.get_writer(uri=path,
                                      format="FFMPEG",
                                      fps=25,
                                      codec="rawvideo",
                                      pixelformat="yuv420p",
                                      macro_block_size=None,
                                      ffmpeg_log_level="error")
            # write the filtered frames to avi file
            for evid in np.arange(len(ds)):
                # skip frames that were filtered out
                if not ds._filter[evid]:
                    continue
                try:
                    image = ds["image"][evid]
                except:
                    warnings.warn("Could not read image {}!".format(evid))
                    continue
                else:
                    if np.isnan(image[0,0]):
                        # This is a nan-valued image
                        image = np.zeros_like(image, dtype=np.uint8)
                # Convert image to RGB
                image = image.reshape(image.shape[0], image.shape[1], 1)
                image = np.repeat(image, 3, axis=2)
                vout.append_data(image)
        else:
            msg="No image data to export: dataset {} !".format(ds.title)
            raise OSError(msg)


    def fcs(self, path, features, filtered=True, override=False):
        """Export the data of an RT-DC dataset to an .fcs file
        
        Parameters
        ----------
        mm: instance of dclab.RTDCBase
            The data set that will be exported.
        path : str
            Path to a .tsv file. The ending .tsv is added automatically.
        features : list of str
            The features in the resulting .tsv file. These are strings
            that are defined in `dclab.definitions.feature_names`, e.g.
            "area_cvx", "deform", "frame", "fl1_max", "aspect".
        filtered : bool
            If set to ``True``, only the filtered data (index in ds._filter)
            are used.
        override : bool
            If set to ``True``, an existing file ``path`` will be overridden.
            If set to ``False``, an ``OSError`` will be raised.
        """
        features = [ c.lower() for c in features ]
        ds = self.rtdc_ds

        # Make sure that path ends with .fcs
        if not path.endswith(".fcs"):
            path += ".fcs"
        # Check if file already exist
        if not override and os.path.exists(path):
            raise OSError("File already exists: {}\n".format(
                                    path.encode("ascii", "ignore"))+
                          "Please use the `override=True` option.")
        # Check that features are in dfn.feature_names
        for c in features:
            if c not in dfn.feature_names:
                raise ValueError("Unknown feature name {}".format(c))
        
        # Collect the header
        chn_names = [ dfn.feature_name2label[c] for c in features ]
    
        # Collect the data
        if filtered:
            data = [ ds[c][ds._filter] for c in features ]
        else:
            data = [ ds[c] for c in features ]
        
        data = np.array(data).transpose()
        fcswrite.write_fcs(filename=path,
                           chn_names=chn_names,
                           data=data)


    def tsv(self, path, features, filtered=True, override=False):
        """ Export the data of the current instance to a .tsv file
        
        Parameters
        ----------
        path : str
            Path to a .tsv file. The ending .tsv is added automatically.
        features : list of str
            The features in the resulting .tsv file. These are strings
            that are defined in `dclab.definitions.feature_names`, e.g.
            "area_cvx", "deform", "frame", "fl1_max", "aspect".
        filtered : bool
            If set to ``True``, only the filtered data (index in ds._filter)
            are used.
        override : bool
            If set to ``True``, an existing file ``path`` will be overridden.
            If set to ``False``, an ``OSError`` will be raised.
        """
        features = [ c.lower() for c in features ]
        ds = self.rtdc_ds
        # Make sure that path ends with .tsv
        if not path.endswith(".tsv"):
            path += ".tsv"
        # Check if file already exist
        if not override and os.path.exists(path):
            raise OSError("File already exists: {}\n".format(
                                    path.encode("ascii", "ignore"))+
                          "Please use the `override=True` option.")
        # Check that features are in dfn.feature_names
        for c in features:
            if c not in dfn.feature_names:
                raise ValueError("Unknown feature name {}".format(c))
        
        # Open file
        with io.open(path, "w") as fd:
            # write header
            header1 = "\t".join([ c for c in features ])
            fd.write("# "+header1+"\n")
            header2 = "\t".join([ dfn.feature_name2label[c] for c in features ])
            fd.write("# "+header2+"\n")

        with open(path, "ab") as fd:
            # write data
            if filtered:
                data = [ ds[c][ds._filter] for c in features ]
            else:
                data = [ ds[c] for c in features ]
            
            np.savetxt(fd,
                       np.array(data).transpose(),
                       fmt=str("%.10e"),
                       delimiter="\t")
