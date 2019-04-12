#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Export RT-DC measurement data"""
from __future__ import division, print_function, unicode_literals

import pathlib
import warnings

import imageio
import fcswrite
import numpy as np

from .. import definitions as dfn
from .write_hdf5 import write


class NoImageWarning(UserWarning):
    pass


class Export(object):
    def __init__(self, rtdc_ds):
        """Export functionalities for RT-DC datasets"""
        self.rtdc_ds = rtdc_ds

    def avi(self, path, filtered=True, override=False):
        """Exports filtered event images to an avi file

        Parameters
        ----------
        path: str
            Path to a .tsv file. The ending .tsv is added automatically.
        filtered: bool
            If set to `True`, only the filtered data (index in ds._filter)
            are used.
        override: bool
            If set to `True`, an existing file ``path`` will be overridden.
            If set to `False`, raises `OSError` if ``path`` exists.

        Notes
        -----
        Raises OSError if current dataset does not contain image data
        """
        path = pathlib.Path(path)
        ds = self.rtdc_ds
        # Make sure that path ends with .avi
        if path.suffix != ".avi":
            path = path.with_name(path.name + ".avi")
        # Check if file already exist
        if not override and path.exists():
            raise OSError("File already exists: {}\n".format(
                str(path).encode("ascii", "ignore")) +
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
                if filtered and not ds._filter[evid]:
                    continue
                try:
                    image = ds["image"][evid]
                except BaseException:
                    warnings.warn("Could not read image {}!".format(evid),
                                  NoImageWarning)
                    continue
                else:
                    if np.isnan(image[0, 0]):
                        # This is a nan-valued image
                        image = np.zeros_like(image, dtype=np.uint8)
                # Convert image to RGB
                image = image.reshape(image.shape[0], image.shape[1], 1)
                image = np.repeat(image, 3, axis=2)
                vout.append_data(image)
        else:
            msg = "No image data to export: dataset {} !".format(ds.title)
            raise OSError(msg)

    def fcs(self, path, features, filtered=True, override=False):
        """Export the data of an RT-DC dataset to an .fcs file

        Parameters
        ----------
        mm: instance of dclab.RTDCBase
            The dataset that will be exported.
        path: str
            Path to a .tsv file. The ending .tsv is added automatically.
        features: list of str
            The features in the resulting .tsv file. These are strings
            that are defined in `dclab.definitions.scalar_feature_names`,
            e.g. "area_cvx", "deform", "frame", "fl1_max", "aspect".
        filtered: bool
            If set to `True`, only the filtered data (index in ds._filter)
            are used.
        override: bool
            If set to `True`, an existing file ``path`` will be overridden.
            If set to `False`, raises `OSError` if ``path`` exists.

        Notes
        -----
        Due to incompatibility with the .fcs file format, all events with
        NaN-valued features are not exported.
        """
        features = [c.lower() for c in features]
        ds = self.rtdc_ds

        path = pathlib.Path(path)
        # Make sure that path ends with .fcs
        if path.suffix != ".fcs":
            path = path.with_name(path.name + ".fcs")
        # Check if file already exist
        if not override and path.exists():
            raise OSError("File already exists: {}\n".format(
                str(path).encode("ascii", "ignore")) +
                "Please use the `override=True` option.")
        # Check that features are in dfn.scalar_feature_names
        for c in features:
            if c not in dfn.scalar_feature_names:
                msg = "Unknown or unsupported feature name: {}".format(c)
                raise ValueError(msg)

        # Collect the header
        chn_names = [dfn.feature_name2label[c] for c in features]

        # Collect the data
        if filtered:
            data = [ds[c][ds._filter] for c in features]
        else:
            data = [ds[c] for c in features]

        data = np.array(data).transpose()
        fcswrite.write_fcs(filename=str(path),
                           chn_names=chn_names,
                           data=data)

    def hdf5(self, path, features, filtered=True, override=False,
             compression="gzip"):
        """Export the data of the current instance to an HDF5 file

        Parameters
        ----------
        path: str
            Path to an .rtdc file. The ending .rtdc is added
            automatically.
        features: list of str
            The features in the resulting .tsv file. These are strings
            that are defined in `dclab.definitions.feature_names`, e.g.
            "area_cvx", "deform", "frame", "fl1_max", "image".
        filtered: bool
            If set to `True`, only the filtered data (index in ds._filter)
            are used.
        override: bool
            If set to `True`, an existing file ``path`` will be overridden.
            If set to `False`, raises `OSError` if ``path`` exists.
        compression: str or None
            Compression method for "contour", "image", and "trace" data
            as well as logs; one of [None, "lzf", "gzip", "szip"].
        """
        path = pathlib.Path(path)
        # Make sure that path ends with .rtdc
        if not path.suffix == ".rtdc":
            path = path.parent / (path.name + ".rtdc")
        # Check if file already exist
        if not override and path.exists():
            raise OSError("File already exists: {}\n".format(path)
                          + "Please use the `override=True` option.")
        elif path.exists():
            path.unlink()

        meta = {}

        # only export configuration meta data (no user-defined config)
        for sec in dfn.CFG_METADATA:
            if sec in ["fmt_tdms"]:
                # ignored sections
                continue
            if sec in self.rtdc_ds.config:
                meta[sec] = self.rtdc_ds.config[sec].copy()

        if filtered:
            filtarr = self.rtdc_ds.filter.all
            nev = np.sum(filtarr)
        else:
            nev = len(self.rtdc_ds)
            filtarr = np.ones(nev, dtype=bool)

        # update number of events
        meta["experiment"]["event count"] = nev

        # write meta data
        with write(path_or_h5file=path, meta=meta, mode="append") as h5obj:
            # write each feature individually
            for feat in features:
                # event-wise, because
                # - tdms-based datasets don't allow indexing with numpy
                # - there might be memory issues
                if feat == "contour":
                    cont_list = []
                    cmax = 0
                    for ii in range(len(self.rtdc_ds)):
                        if filtarr[ii]:
                            dat = self.rtdc_ds["contour"][ii]
                            cont_list.append(dat)
                            cmax = max(cmax, dat.max())
                    write(h5obj,
                          data={"contour": cont_list},
                          mode="append",
                          compression=compression)
                elif feat in ["mask", "image"]:
                    # store image stacks (reduced file size and save time)
                    m = 64
                    im0 = self.rtdc_ds[feat][0]
                    imstack = np.zeros((m, im0.shape[0], im0.shape[1]),
                                       dtype=im0.dtype)
                    jj = 0
                    for ii in range(len(self.rtdc_ds)):
                        if filtarr[ii]:
                            dat = self.rtdc_ds[feat][ii]
                            imstack[jj] = dat
                            if (jj + 1) % m == 0:
                                jj = 0
                                write(h5obj,
                                      data={feat: imstack},
                                      mode="append",
                                      compression=compression)
                            else:
                                jj += 1
                    # write rest
                    if jj:
                        write(h5obj,
                              data={feat: imstack[:jj, :, :]},
                              mode="append",
                              compression=compression)
                elif feat == "trace":
                    for tr in self.rtdc_ds["trace"].keys():
                        tr0 = self.rtdc_ds["trace"][tr][0]
                        trdat = np.zeros((nev, tr0.size), dtype=tr0.dtype)
                        jj = 0
                        for ii in range(len(self.rtdc_ds)):
                            if filtarr[ii]:
                                trdat[jj] = self.rtdc_ds["trace"][tr][ii]
                                jj += 1
                        write(h5obj,
                              data={"trace": {tr: trdat}},
                              mode="append",
                              compression=compression)
                elif feat == "index" and filtered:
                    # re-enumerate data index feature (filtered data)
                    write(h5obj,
                          data={"index": np.arange(1, nev+1)},
                          mode="append")
                else:
                    write(h5obj,
                          data={feat: self.rtdc_ds[feat][filtarr]},
                          mode="append")

    def tsv(self, path, features, filtered=True, override=False):
        """Export the data of the current instance to a .tsv file

        Parameters
        ----------
        path: str
            Path to a .tsv file. The ending .tsv is added automatically.
        features: list of str
            The features in the resulting .tsv file. These are strings
            that are defined in `dclab.definitions.scalar_feature_names`,
            e.g. "area_cvx", "deform", "frame", "fl1_max", "aspect".
        filtered: bool
            If set to `True`, only the filtered data (index in ds._filter)
            are used.
        override: bool
            If set to `True`, an existing file ``path`` will be overridden.
            If set to `False`, raises `OSError` if ``path`` exists.
        """
        features = [c.lower() for c in features]
        path = pathlib.Path(path)
        ds = self.rtdc_ds
        # Make sure that path ends with .tsv
        if path.suffix != ".tsv":
            path = path.with_name(path.name + ".tsv")
        # Check if file already exist
        if not override and path.exists():
            raise OSError("File already exists: {}\n".format(
                str(path).encode("ascii", "ignore")) +
                "Please use the `override=True` option.")
        # Check that features are in dfn.scalar_feature_names
        for c in features:
            if c not in dfn.scalar_feature_names:
                raise ValueError("Unknown feature name {}".format(c))

        # Open file
        with path.open("w") as fd:
            # write header
            header1 = "\t".join([c for c in features])
            fd.write("# "+header1+"\n")
            header2 = "\t".join([dfn.feature_name2label[c] for c in features])
            fd.write("# "+header2+"\n")

        with path.open("ab") as fd:
            # write data
            if filtered:
                data = [ds[c][ds._filter] for c in features]
            else:
                data = [ds[c] for c in features]

            np.savetxt(fd,
                       np.array(data).transpose(),
                       fmt=str("%.10e"),
                       delimiter="\t")
