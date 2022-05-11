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
from .writer import RTDCWriter, CHUNK_SIZE


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

    def hdf5(self, path, features=None, filtered=True, override=False,
             compression="gzip", skip_checks=False):
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
            Defaults to `self.rtdc_ds.features_innate`.
        filtered: bool
            If set to `True`, only the filtered data
            (index in ds.filter.all) are used.
        override: bool
            If set to `True`, an existing file ``path`` will be overridden.
            If set to `False`, raises `OSError` if ``path`` exists.
        compression: str or None
            Compression method for e.g. "contour", "image", and "trace" data
            as well as logs; one of [None, "lzf", "gzip", "szip"].
        skip_checks: bool
            Disable checking whether all features have the same length.
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

        if features is None:
            features = self.rtdc_ds.features_innate

        # decide which metadata to export
        meta = {}
        # only cfg metadata (no analysis metadata)
        for sec in dfn.CFG_METADATA:
            if sec in self.rtdc_ds.config:
                meta[sec] = self.rtdc_ds.config[sec].copy()
        # add user-defined metadata
        if "user" in self.rtdc_ds.config:
            meta["user"] = self.rtdc_ds.config["user"].copy()

        if filtered:
            filtarr = self.rtdc_ds.filter.all
        else:
            filtarr = None

        if not skip_checks:
            # check that all features have same length and use the smallest
            # common length
            lengths = []
            for feat in features:
                if feat == "trace":
                    for tr in list(self.rtdc_ds["trace"].keys()):
                        lengths.append(len(self.rtdc_ds["trace"][tr]))
                else:
                    lengths.append(len(self.rtdc_ds[feat]))
            lmin = np.min(lengths)
            lmax = np.max(lengths)
            if lmin != lmax:
                if filtarr is None:
                    # we are forced to do filtering
                    filtarr = np.ones(len(self.rtdc_ds), dtype=bool)
                filtarr[lmin:] = False
                warnings.warn(
                    "Not all features have the same length! Limiting output "
                    + f"event count to {lmin} (max {lmax}) in '{lmin}'.",
                    LimitingExportSizeWarning)

        # Perform actual export
        with RTDCWriter(path, mode="append", compression=compression) as hw:
            # write meta data
            hw.store_metadata(meta)
            # write each feature individually
            for feat in features:
                if filtarr is None:
                    # We do not have to filter and can be fast
                    hw.store_feature(feat=feat, data=self.rtdc_ds[feat])
                else:
                    # We have to filter and will be slower
                    store_filtered_feature(rtdc_writer=hw,
                                           feat=feat,
                                           data=self.rtdc_ds[feat],
                                           filtarr=filtarr)

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


def yield_filtered_array_stacks(data, indices):
    """Generator returning chunks with the filtered feature data

    Parameters
    ----------
    data: np.ndarray or h5py.Dataset
        The full, unfiltered input feature data.
    indices: np.ndarray or list
        The indices in data (first axis) that should be written
        to the chunks returned by this generator.

    Notes
    -----
    This method works with any feature dimension (e.g. it
    works for image (2D) data and for trace data (1D)). It
    is just important that `data` is indexable using integers
    and that the events in `data` all have the same shape.
    The dtype of the returned chunks is determined by the first
    item in `data`.
    """
    # assemble filtered image stacks
    data0 = data[0]
    chunk_shape = tuple([CHUNK_SIZE] + list(data0.shape))
    chunk = np.zeros(chunk_shape, dtype=data0.dtype)
    jj = 0
    for ii in indices:
        chunk[jj] = data[ii]
        if (jj + 1) % CHUNK_SIZE == 0:
            jj = 0
            yield chunk
        else:
            jj += 1
    # yield remainder
    if jj:
        yield chunk[:jj]


def store_filtered_feature(rtdc_writer, feat, data, filtarr):
    """Append filtered feature data to an HDF5 file

    Parameters
    ----------
    rtdc_writer: dclab.rtdc_dataset.writer.RTDCWriter
        an open writer object
    feat: str
        feature name
    data: object or list or np.ndarray or dict
        feature data
    filtarr: boolean np.ndarray
        filtering array (same as RTDCBase.filter.all)

    Notes
    -----
    This code is somewhat redundant the the code of RTDCWriter.
    """
    indices = np.where(filtarr)[0]
    hw = rtdc_writer
    if not hw.mode == "append":
        raise ValueError("The `rtdc_writer` object must be created with"
                         + f"`mode='append'`, got '{hw.mode}' for '{hw}'!")
    # event-wise, because
    # - tdms-based datasets don't allow indexing with numpy
    # - there might be memory issues
    if feat == "contour":
        for ii in indices:
            hw.store_feature("contour", data[ii])
    elif feat in ["mask", "image", "image_bg"]:
        # assemble filtered image stacks
        for imstack in yield_filtered_array_stacks(data, indices):
            hw.store_feature(feat, imstack)
    elif feat == "trace":
        # assemble filtered trace stacks
        for tr in data.keys():
            for trstack in yield_filtered_array_stacks(data[tr], indices):
                hw.store_feature("trace", {tr: trstack})
    elif dfn.scalar_feature_exists(feat):
        hw.store_feature(feat, data[filtarr])
    else:
        # Special case of plugin or temporary features.
        shape = data[0].shape
        for dstack in yield_filtered_array_stacks(data, indices):
            hw.store_feature(feat, dstack, shape=shape)


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
        Do not use! Please use `dclab.cli.task_join.join` instead.

    Notes
    -----
    Please update the "experiment::event count" attribute manually.
    You may use
    :func:`dclab.rtdc_dataset.writer.RTDCWriter.rectify_metadata`
    for that or use the `RTDCWriter` context manager where it is
    automatically run during `__exit__`.
    """
    # optional array for filtering events
    if filtarr is None:
        filtarr = np.ones(len(rtdc_ds), dtype=bool)
        no_filter = True
    else:
        no_filter = False

    warnings.warn("`hdf5_append` is deptecated; please use "
                  " the dclab.RTDCWriter context manager or the "
                  " export.store_filtered_feature function.",
                  DeprecationWarning)

    if time_offset != 0:
        raise ValueError("Setting `time_offset` not supported anymore! "
                         "Please use `dclab.cli.task_join.join` instead.")

    # writer instance
    hw = RTDCWriter(h5obj, mode="append", compression=compression)
    if no_filter:
        hw.store_feature(feat, rtdc_ds[feat])
    else:
        store_filtered_feature(rtdc_writer=hw,
                               feat=feat,
                               data=rtdc_ds[feat],
                               filtarr=filtarr)


def hdf5_autocomplete_config(path_or_h5obj):
    """"Autocomplete the configuration of the RTDC-measurement

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
    warnings.warn("`hdf5_autocomplete_config` is deptecated; please use "
                  " the dclab.RTDCWriter context manager or the "
                  " dclab.RTDCWriter.rectify_metadata function.",
                  DeprecationWarning)
    if not isinstance(path_or_h5obj, h5py.File):
        close = True
    else:
        close = False

    hw = RTDCWriter(path_or_h5obj, mode="append")
    hw.rectify_metadata()

    if close:
        path_or_h5obj.close()
