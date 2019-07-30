#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Export RT-DC measurement data"""
from __future__ import division, print_function, unicode_literals

import pathlib
import warnings

import h5py

from ..compat import PyImportError

try:
    import imageio
except PyImportError:
    IMAGEIO_AVAILABLE = False
else:
    IMAGEIO_AVAILABLE = True

try:
    import fcswrite
except PyImportError:
    FCSWRITE_AVAILABLE = False
else:
    FCSWRITE_AVAILABLE = True

import numpy as np

from .. import definitions as dfn
from .write_hdf5 import write


class NoImageWarning(UserWarning):
    pass


class LimitingExportSizeWarning(UserWarning):
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
            Path to a .avi file. The ending .avi is added automatically.
        filtered: bool
            If set to `True`, only the filtered data
            (index in ds.filter.all) are used.
        override: bool
            If set to `True`, an existing file ``path`` will be overridden.
            If set to `False`, raises `OSError` if ``path`` exists.

        Notes
        -----
        Raises OSError if current dataset does not contain image data
        """
        if not IMAGEIO_AVAILABLE:
            raise PyImportError("Package `imageio` required for avi export!")
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
            Path to an .fcs file. The ending .fcs is added automatically.
        features: list of str
            The features in the resulting .fcs file. These are strings
            that are defined in `dclab.definitions.scalar_feature_names`,
            e.g. "area_cvx", "deform", "frame", "fl1_max", "aspect".
        filtered: bool
            If set to `True`, only the filtered data
            (index in ds.filter.all) are used.
        override: bool
            If set to `True`, an existing file ``path`` will be overridden.
            If set to `False`, raises `OSError` if ``path`` exists.

        Notes
        -----
        Due to incompatibility with the .fcs file format, all events with
        NaN-valued features are not exported.
        """
        if not FCSWRITE_AVAILABLE:
            raise PyImportError("Package `fcswrite` required for fcs export!")
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
            The features in the resulting .rtdc file. These are strings
            that are defined in `dclab.definitions.feature_names`, e.g.
            "area_cvx", "deform", "frame", "fl1_max", "image".
        filtered: bool
            If set to `True`, only the filtered data
            (index in ds.filter.all) are used.
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
        # Check if file already exists
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
        else:
            filtarr = np.ones(len(self.rtdc_ds), dtype=bool)

        # check that all features have same length and use smallest
        # common length
        lengths = []
        for feat in features:
            if feat == "trace":
                for tr in list(self.rtdc_ds["trace"].keys()):
                    lengths.append(len(self.rtdc_ds["trace"][tr]))
            else:
                lengths.append(len(self.rtdc_ds[feat]))
        if not np.all(lengths == lengths[0]):
            lmin = np.min(lengths)
            nev_bef = np.sum(filtarr)
            filtarr[lmin:] = False
            nev_aft = np.sum(filtarr)
            if nev_bef != nev_aft:
                warnings.warn(
                    "Not all features have the same length! "
                    + "Limiting output event count to {} ".format(lmin)
                    + "in '{}'.".format(path), LimitingExportSizeWarning)

        # write meta data
        with write(path_or_h5file=path, meta=meta, mode="append") as h5obj:
            # write each feature individually
            for feat in features:
                hdf5_append(h5obj=h5obj,
                            rtdc_ds=self.rtdc_ds,
                            feat=feat,
                            compression=compression,
                            filtarr=filtarr)
            # update configuration
            hdf5_autocomplete_config(h5obj)

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
            If set to `True`, only the filtered data
            (index in ds.filter.all) are used.
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


def hdf5_append(h5obj, rtdc_ds, feat, compression, filtarr=None):
    """Append feature data to an HDF5 file

    Parameters
    ----------
    h5obj: h5py.File
        Opened HDF5 file
    rtdc_ds: dclab.rtdc_dataset.RTDCBase
        Instance from which to obtain the data
    feat: str
        Valid feature name in `rtdc_ds`
    compression: str or None
        Compression method for "contour", "image", and "trace" data
        as well as logs; one of [None, "lzf", "gzip", "szip"].
    filtarr: None or 1d boolean np.ndarray
        Optional boolean array used for filtering. If set to
        `None`, all events are saved.

    Notes
    -----
    Please update the "experiment::event count" attribute manually.
    You may use :func:`hdf5_autocomplete_config` for that.
    """
    # optional array for filtering events
    if filtarr is None:
        filtarr = np.ones(len(rtdc_ds), dtype=bool)
    # total number of new events
    nev = np.sum(filtarr)
    # event-wise, because
    # - tdms-based datasets don't allow indexing with numpy
    # - there might be memory issues
    if feat == "contour":
        cont_list = []
        cmax = 0
        for ii in range(len(rtdc_ds)):
            if filtarr[ii]:
                dat = rtdc_ds["contour"][ii]
                cont_list.append(dat)
                cmax = max(cmax, dat.max())
        write(h5obj,
              data={"contour": cont_list},
              mode="append",
              compression=compression)
    elif feat in ["mask", "image"]:
        # store image stacks (reduced file size and save time)
        m = 64
        im0 = rtdc_ds[feat][0]
        imstack = np.zeros((m, im0.shape[0], im0.shape[1]),
                           dtype=im0.dtype)
        jj = 0
        for ii in range(len(rtdc_ds)):
            if filtarr[ii]:
                dat = rtdc_ds[feat][ii]
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
        for tr in rtdc_ds["trace"].keys():
            tr0 = rtdc_ds["trace"][tr][0]
            trdat = np.zeros((nev, tr0.size), dtype=tr0.dtype)
            jj = 0
            for ii in range(len(rtdc_ds)):
                if filtarr[ii]:
                    trdat[jj] = rtdc_ds["trace"][tr][ii]
                    jj += 1
            write(h5obj,
                  data={"trace": {tr: trdat}},
                  mode="append",
                  compression=compression)
    elif feat == "index":
        # re-enumerate data index feature (filtered data)
        if "events/index" in h5obj:
            nev0 = len(h5obj["events/index"])
        else:
            nev0 = 0
        write(h5obj,
              data={"index": np.arange(nev0+1, nev0+nev+1)},
              mode="append",
              compression=compression)
    else:
        write(h5obj,
              data={feat: rtdc_ds[feat][filtarr]},
              mode="append",
              compression=compression)


def hdf5_autocomplete_config(path_or_h5obj):
    """"Autocompletes the configuration of the RTDC-measurement

    The following configuration keys are updated:

    - experiment:event count
    - fluorescence:samples per event

    The following configuration keys are added if not present:

    - fluorescence:channel count

    Parameters
    ----------
    path: pathlib.Path or str or h5py.File
        Path to or opened RT-DC measurement

    """
    if not isinstance(path_or_h5obj, h5py.File):
        close = True
        h5obj = h5py.File(path_or_h5obj, "a")
    else:
        close = False
        h5obj = path_or_h5obj

    # set event count
    feats = sorted(h5obj["events"].keys())
    if feats:
        h5obj.attrs["experiment:event count"] = len(h5obj["events"][feats[0]])
    else:
        raise ValueError("No features in '{}'!".format(path_or_h5obj))
    # set samples per event
    if "trace" in feats:
        traces = list(h5obj["events"]["trace"].keys())
        trsize = h5obj["events"]["trace"][traces[0]].shape[1]
        h5obj.attrs["fluorescence:samples per event"] = trsize
    # set channel count
    chcount = sum(["fl1_max" in feats, "fl2_max" in feats, "fl3_max" in feats])
    if chcount:
        if "fluorescence:channel count" not in h5obj.attrs:
            h5obj.attrs["fluorescence:channel count"] = chcount

    if close:
        h5obj.close()
