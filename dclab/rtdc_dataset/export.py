"""Export RT-DC measurement data"""

import codecs
import pathlib
import warnings

import h5py

try:
    import imageio
except ModuleNotFoundError:
    IMAGEIO_AVAILABLE = False
else:
    IMAGEIO_AVAILABLE = True

try:
    import fcswrite
except ModuleNotFoundError:
    FCSWRITE_AVAILABLE = False
else:
    FCSWRITE_AVAILABLE = True

import numpy as np

from .. import definitions as dfn
from .._version import version
from .write_hdf5 import write, CHUNK_SIZE


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
            raise ModuleNotFoundError(
                "Package `imageio` required for avi export!")
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
                if filtered and not ds.filter.all[evid]:
                    continue
                image = ds["image"][evid]
                # Convert image to RGB
                image = image.reshape(image.shape[0], image.shape[1], 1)
                image = np.repeat(image, 3, axis=2)
                vout.append_data(image)
        else:
            msg = "No image data to export: dataset {} !".format(ds.title)
            raise OSError(msg)

    def fcs(self, path, features, meta_data=None, filtered=True,
            override=False):
        """Export the data of an RT-DC dataset to an .fcs file

        Parameters
        ----------
        path: str
            Path to an .fcs file. The ending .fcs is added automatically.
        features: list of str
            The features in the resulting .fcs file. These are strings
            that are defined by `dclab.definitions.scalar_feature_exists`,
            e.g. "area_cvx", "deform", "frame", "fl1_max", "aspect".
        meta_data: dict
            User-defined, optional key-value pairs that are stored
            in the primary TEXT segment of the FCS file; the version
            of dclab is stored there by default
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
        if meta_data is None:
            meta_data = {}
        if not FCSWRITE_AVAILABLE:
            raise ModuleNotFoundError(
                "Package `fcswrite` required for fcs export!")
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
        # Check that features are valid
        for c in features:
            if c not in ds.features_scalar:
                msg = "Invalid feature name: {}".format(c)
                raise ValueError(msg)

        # Collect the header
        chn_names = [dfn.get_feature_label(c, rtdc_ds=ds) for c in features]

        # Collect the data
        if filtered:
            data = [ds[c][ds.filter.all] for c in features]
        else:
            data = [ds[c] for c in features]

        data = np.array(data).transpose()
        meta_data["dclab version"] = version
        fcswrite.write_fcs(filename=str(path),
                           chn_names=chn_names,
                           data=data,
                           text_kw_pr=meta_data,
                           )

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
            that are defined by `dclab.definitions.feature_exists`, e.g.
            "area_cvx", "deform", "frame", "fl1_max", "image".
        filtered: bool
            If set to `True`, only the filtered data
            (index in ds.filter.all) are used.
        override: bool
            If set to `True`, an existing file ``path`` will be overridden.
            If set to `False`, raises `OSError` if ``path`` exists.
        compression: str or None
            Compression method for e.g. "contour", "image", and "trace" data
            as well as logs; one of [None, "lzf", "gzip", "szip"].
        """
        path = pathlib.Path(path)
        # Make sure that path ends with .rtdc
        if path.suffix not in [".rtdc", ".rtdc~"]:
            path = path.parent / (path.name + ".rtdc")
        # Check if file already exists
        if not override and path.exists():
            raise OSError("File already exists: {}\n".format(path)
                          + "Please use the `override=True` option.")
        elif path.exists():
            path.unlink()

        meta = {}

        # export configuration meta data
        for sec in dfn.CFG_METADATA:
            if sec in ["fmt_tdms"]:
                # ignored sections
                continue
            if sec in self.rtdc_ds.config:
                meta[sec] = self.rtdc_ds.config[sec].copy()
        # add user-defined metadata
        if "user" in self.rtdc_ds.config:
            meta["user"] = self.rtdc_ds.config["user"].copy()

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
            lmax = np.max(lengths)
            nev_bef = np.sum(filtarr)
            filtarr[lmin:] = False
            nev_aft = np.sum(filtarr)
            if nev_bef != nev_aft:
                warnings.warn(
                    "Not all features have the same length! "
                    + "Limiting output event count to {} ".format(lmin)
                    + "(max {}) in '{}'.".format(lmax, path),
                    LimitingExportSizeWarning)

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

    def tsv(self, path, features, meta_data=None, filtered=True,
            override=False):
        """Export the data of the current instance to a .tsv file

        Parameters
        ----------
        path: str
            Path to a .tsv file. The ending .tsv is added automatically.
        features: list of str
            The features in the resulting .tsv file. These are strings
            that are defined by `dclab.definitions.scalar_feature_exists`,
            e.g. "area_cvx", "deform", "frame", "fl1_max", "aspect".
        meta_data: dict
            User-defined, optional key-value pairs that are stored
            at the beginning of the tsv file - one key-value pair is
            stored per line which starts with a hash. The version of
            dclab is stored there by default.
        filtered: bool
            If set to `True`, only the filtered data
            (index in ds.filter.all) are used.
        override: bool
            If set to `True`, an existing file ``path`` will be overridden.
            If set to `False`, raises `OSError` if ``path`` exists.
        """
        if meta_data is None:
            meta_data = {}
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
        # Check that features exist
        for c in features:
            if c not in ds.features_scalar:
                raise ValueError("Invalid feature name {}".format(c))
        meta_data["dclab version"] = version
        # Write BOM header
        with path.open("wb") as fd:
            fd.write(codecs.BOM_UTF8)
        # Open file
        with path.open("a", encoding="utf-8") as fd:
            # write meta data
            for key in sorted(meta_data.keys()):
                fd.write("# {}: {}\n".format(key, meta_data[key]))
            fd.write("#\n")
            # write header
            header1 = "\t".join([c for c in features])
            fd.write("# "+header1+"\n")
            labels = [dfn.get_feature_label(c, rtdc_ds=ds) for c in features]
            header2 = "\t".join(labels)
            fd.write("# "+header2+"\n")

        with path.open("ab") as fd:
            # write data
            if filtered:
                data = [ds[c][ds.filter.all] for c in features]
            else:
                data = [ds[c] for c in features]

            np.savetxt(fd,
                       np.array(data).transpose(),
                       fmt=str("%.10e"),
                       delimiter="\t")


def hdf5_append(h5obj, rtdc_ds, feat, compression, filtarr=None,
                time_offset=0):
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
    time_offset: float
        This value will be added to the "time" and "frame" features
        (used for joining multiple measurements)

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
        for ii in range(len(rtdc_ds)):
            if filtarr[ii]:
                cont_list.append(rtdc_ds["contour"][ii])
        write(h5obj,
              data={"contour": cont_list},
              mode="append",
              compression=compression)
    elif feat in ["mask", "image", "image_bg"]:
        # store image stacks (reduces file size, memory usage, and saves time)
        im0 = rtdc_ds[feat][0]
        imstack = np.zeros((CHUNK_SIZE, im0.shape[0], im0.shape[1]),
                           dtype=im0.dtype)
        jj = 0
        for ii in range(len(rtdc_ds)):
            if filtarr[ii]:
                imstack[jj] = rtdc_ds[feat][ii]
                if (jj + 1) % CHUNK_SIZE == 0:
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
        # store trace stacks (reduces file size, memory usage, and saves time)
        for tr in rtdc_ds["trace"].keys():
            tr0 = rtdc_ds["trace"][tr][0]
            trstack = np.zeros((CHUNK_SIZE, len(tr0)), dtype=tr0.dtype)
            jj = 0
            for ii in range(len(rtdc_ds)):
                if filtarr[ii]:
                    trstack[jj] = rtdc_ds["trace"][tr][ii]
                    if (jj + 1) % CHUNK_SIZE == 0:
                        jj = 0
                        write(h5obj,
                              data={"trace": {tr: trstack}},
                              mode="append",
                              compression=compression)
                    else:
                        jj += 1
            # write rest
            if jj:
                write(h5obj,
                      data={"trace": {tr: trstack[:jj, :]}},
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
    elif feat == "index_online":
        if "events/index_online" in h5obj:
            # index_online is usually larger than index
            ido0 = h5obj["events/index_online"][-1] + 1
        else:
            ido0 = 0
        write(h5obj,
              data={"index_online": rtdc_ds["index_online"][filtarr] + ido0},
              mode="append",
              compression=compression)
    elif feat == "time":
        write(h5obj,
              data={"time": rtdc_ds["time"][filtarr] + time_offset},
              mode="append",
              compression=compression)
    elif feat == "frame":
        if time_offset != 0:
            # Only get the frame rate when we actually need it.
            fr = rtdc_ds.config["imaging"]["frame rate"]
        else:
            fr = 0
        frame_offset = time_offset * fr
        write(h5obj,
              data={"frame": rtdc_ds["frame"][filtarr] + frame_offset},
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
    - imaging: roi size x (if image or mask is given)
    - imaging: roi size y (if image or mask is given)

    The following configuration keys are added if not present:

    - fluorescence:channel count

    Parameters
    ----------
    path_or_h5obj: pathlib.Path or str or h5py.File
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

    # set roi size x/y
    if "image" in h5obj["events"]:
        shape = h5obj["events"]["image"][0].shape
    elif "mask" in h5obj["events"]:
        shape = h5obj["events"]["mask"][0].shape
    else:
        shape = None
    if shape is not None:
        # update shape
        h5obj.attrs["imaging:roi size x"] = shape[1]
        h5obj.attrs["imaging:roi size y"] = shape[0]

    if close:
        h5obj.close()
