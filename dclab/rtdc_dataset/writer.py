from __future__ import annotations

from collections.abc import Mapping
import copy
import json
import os
import pathlib
from typing import Dict, List, Literal, Tuple
import warnings

import h5py
import hdf5plugin
import numpy as np

from .. import definitions as dfn
from ..util import hashobj
from .._version import version

from .feat_anc_plugin import PlugInFeature

#: DEPRECATED (use `CHUNK_SIZE_BYTES` instead)
CHUNK_SIZE = 100

#: Chunks size in bytes for storing HDF5 datasets
CHUNK_SIZE_BYTES = 1024**2  # 1MiB


class RTDCWriter:
    def __init__(self,
                 path_or_h5file: str | pathlib.Path | h5py.File,
                 mode: Literal['append', 'replace', 'reset'] = "append",
                 compression_kwargs: Dict | Mapping = None,
                 compression: str = "deprecated"):
        """RT-DC data writer classe

        Parameters
        ----------
        path_or_h5file: str or pathlib.Path or h5py.Group
            Path to an HDF5 file or an HDF5 file opened in write mode
        mode: str
            Defines how the data are stored:

              - "append": append new feature data to existing h5py Datasets
              - "replace": replace existing h5py Datasets with new features
                (used for ancillary feature storage)
              - "reset": do not keep any previous data
        compression_kwargs: dict-like
            Dictionary with the keys "compression" and "compression_opts"
            which are passed to :func:`h5py.H5File.create_dataset`. The
            default is Zstandard compression with the lowest compression
            level `hdf5plugin.Zstd(clevel=1)`. To disable compression, use
            `{"compression": None}`.
        compression: str or None
            Compression method used for data storage;
            one of [None, "lzf", "gzip", "szip"].

            .. deprecated:: 0.43.0
                Use `compression_kwargs` instead.
        """
        if mode not in ["append", "replace", "reset"]:
            raise ValueError(f"Invalid mode '{mode}'!")
        if compression != "deprecated":
            warnings.warn("The `compression` kwarg is deprecated in favor of "
                          "`compression_kwargs`!",
                          DeprecationWarning)
            if compression_kwargs is not None:
                raise ValueError("You may not specify `compression` and "
                                 "`compression_kwargs` at the same time!")
            # be backwards-compatible
            compression_kwargs = {"compression": compression}
        if compression_kwargs is None:
            compression_kwargs = hdf5plugin.Zstd(clevel=1)

        self.mode = mode
        self.compression_kwargs = compression_kwargs
        if isinstance(path_or_h5file, h5py.Group):
            self.owns_path = False
            self.path = pathlib.Path(path_or_h5file.file.filename)
            self.h5file = path_or_h5file
            if mode == "reset":
                raise ValueError("'reset' mode incompatible with h5py.Group!")
        else:
            self.owns_path = True
            self.path = pathlib.Path(path_or_h5file)
            self.h5file = h5py.File(path_or_h5file,
                                    mode=("w" if mode == "reset" else "a"))
        #: unfortunate necessity, as `len(h5py.Group)` can be really slow
        self._group_sizes = {}

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        # close the HDF5 file
        try:
            self.h5file.require_group("events")
            if len(self.h5file["events"]):
                self.rectify_metadata()
            self.version_brand()
        except BaseException:
            raise
        finally:
            # This is guaranteed to run if any exception is raised.
            self.close()

    @staticmethod
    def get_best_nd_chunks(item_shape, item_dtype=np.float64):
        """Return best chunks for HDF5 datasets

        Chunking has performance implications. It’s recommended to keep the
        total size of dataset chunks between 10 KiB and 1 MiB. This number
        defines the maximum chunk size as well as half the maximum cache
        size for each dataset.
        """
        # Note that `np.prod(()) == 1`
        event_size = np.prod(item_shape) * np.dtype(item_dtype).itemsize

        chunk_size = CHUNK_SIZE_BYTES / event_size
        # Set minimum chunk size to 10 so that we can have at least some
        # compression performance.
        chunk_size_int = max(10, int(np.floor(chunk_size)))
        return tuple([chunk_size_int] + list(item_shape))

    def close(self):
        """Close the underlying HDF5 file if a path was given during init"""
        if self.owns_path:
            self.h5file.close()

    def rectify_metadata(self):
        """Autocomplete the metadta of the RTDC-measurement

        The following configuration keys are updated:

        - experiment:event count
        - fluorescence:samples per event
        - imaging: roi size x (if image or mask is given)
        - imaging: roi size y (if image or mask is given)

        The following configuration keys are added if not present:

        - fluorescence:channel count
        """
        # set event count
        feats = sorted(self.h5file.get("events", {}).keys())
        if feats:
            self.h5file.attrs["experiment:event count"] = len(
                self.h5file["events"][feats[0]])
        else:
            raise ValueError(f"No features in '{self.path}'!")

        # ignore empty features in the checks further below
        for feat in feats[:]:  # iterate over a copy of the list
            obj = self.h5file["events"][feat]
            if ((isinstance(feat, h5py.Dataset) and obj.shape[0] == 0)  # ds
                    or len(obj) == 0):  # groups
                feats.remove(feat)

        # set samples per event
        if "trace" in feats:
            traces = list(self.h5file["events"]["trace"].keys())
            trsize = self.h5file["events"]["trace"][traces[0]].shape[1]
            self.h5file.attrs["fluorescence:samples per event"] = trsize

        # set channel count
        chcount = sum(
            ["fl1_max" in feats, "fl2_max" in feats, "fl3_max" in feats])
        if chcount:
            if "fluorescence:channel count" not in self.h5file.attrs:
                self.h5file.attrs["fluorescence:channel count"] = chcount

        # set roi size x/y
        if "image" in feats:
            shape = self.h5file["events"]["image"][0].shape
        elif "mask" in feats:
            shape = self.h5file["events"]["mask"][0].shape
        else:
            shape = None
        if shape is not None:
            # update shape
            self.h5file.attrs["imaging:roi size x"] = shape[1]
            self.h5file.attrs["imaging:roi size y"] = shape[0]

    def store_basin(self,
                    basin_name: str,
                    basin_type: Literal['file', 'internal', 'remote'],
                    basin_format: str,
                    basin_locs: List[str | pathlib.Path],
                    basin_descr: str | None = None,
                    basin_feats: List[str] = None,
                    basin_map: np.ndarray | Tuple[str, np.ndarray] = None,
                    internal_data: Dict | h5py.Group = None,
                    verify: bool = True,
                    ):
        """Write basin information

        Parameters
        ----------
        basin_name: str
            basin name; Names do not have to be unique.
        basin_type: str
            basin type (file or remote); Files are paths accessible by the
            operating system (including e.g. network shares) whereas
            remote locations normally require an active internet connection.
        basin_format: str
            The basin format must match the ``format`` property of an
            :class:`.RTDCBase` subclass (e.g. "hdf5" or "dcor")
        basin_locs: list
            location of the basin as a string or (optionally)
            a ``pathlib.Path``
        basin_descr: str
            optional string describing the basin
        basin_feats: list of str
            list of features this basin provides; You may use this to
            restrict access to features for a specific basin.
        basin_map: np.ndarray or tuple of (str, np.ndarray)
            If this is an integer numpy array, it defines the mapping
            of event indices from the basin dataset to the referring dataset
            (the dataset being written to disk). Normally, the basinmap
            feature used for storing the mapping information is inferred
            from the currently defined basinmap features. However, if you
            are incepting basins, then this might not be sufficient, and you
            have to specify explicitly which basinmap feature to use. In such
            a case, you may specify a tuple `(feature_name, mapping_array)`
            where `feature_name` is the explicit mapping name, e.g.
            `"basinmap3"`.
        internal_data: dict or instance of h5py.Group
            A dictionary or an `h5py.Group` containing the basin data.
            The data are copied to the "basin_events" group, if
            `internal_data` is not an `h5py.Group` in the current HDF5 file.
            This must be specified when storing internal basins, and it
            must not be specified for any other basin type.
        verify: bool
            whether to verify the basin before storing it; You might have
            set this to False if you would like to write a basin that is
            e.g. temporarily not available

        Returns
        -------
        basin_hash: str
            hash of the basin which serves as the name of the HDF5 dataset
            stored in the output file

            .. versionadded:: 0.58.0
        """
        if basin_type == "internal":
            if internal_data is None:
                raise ValueError(
                    "When writing an internal basin, you must specify "
                    "`internal_data` which is either a dictionary of numpy "
                    "arrays or an `h5py.Group` containing the relevant "
                    "datasets.")
            if (isinstance(internal_data, dict)
                    or (isinstance(internal_data, h5py.Group)
                        and internal_data.file != self.h5file)):
                # The data are not yet stored in this HDF5 file
                for feat in basin_feats:
                    igroup = self.h5file.require_group("basin_events")
                    if feat in igroup:
                        raise ValueError(f"The feature '{feat}' already "
                                         f"exists in the 'basin_events' group")
                    self.write_ndarray(group=igroup,
                                       name=feat,
                                       data=internal_data[feat])
                # just override it with the default
                basin_locs = ["basin_events"]
            elif verify:
                # Verify the existence of the data inside this HDF5 file
                if basin_locs != ["basin_events"]:
                    warnings.warn("You specified an uncommon location for "
                                  f"your internal basins: {basin_locs}. "
                                  f"Please use 'basin_events' instead.")
                for feat in basin_feats:
                    if feat not in self.h5file[basin_locs[0]]:
                        raise ValueError(f"Could not find feature '{feat}' in "
                                         f"the group [{basin_locs[0]}]")

        # Expand optional tuple for basin_map
        if isinstance(basin_map, (list, tuple)) and len(basin_map) == 2:
            basin_map_name, basin_map = basin_map
        else:
            basin_map_name = None

        if verify and basin_type in ["file", "remote"]:
            # We have to import this here to avoid circular imports
            from .load import new_dataset
            # Make sure the basin can be opened by dclab, verify its ID
            cur_id = self.h5file.attrs.get("experiment:run identifier")
            for loc in basin_locs:
                with new_dataset(loc) as ds:
                    # We can open the file, which is great.
                    if cur_id:
                        # Compare the IDs.
                        ds_id = ds.get_measurement_identifier()
                        if not (ds_id == cur_id
                                or (basin_map is not None
                                    and cur_id.startswith(ds_id))):
                            raise ValueError(
                                f"Measurement identifier mismatch between "
                                f"{self.path} ({cur_id}) and {loc} ({ds_id})!")
            if basin_feats:
                for feat in basin_feats:
                    if not dfn.feature_exists(feat):
                        raise ValueError(f"Invalid feature: '{feat}'")
            if basin_map is not None:
                if (not isinstance(basin_map, np.ndarray)
                        and basin_map.dtype != np.uint64):
                    raise ValueError(
                        "The array specified in `basin_map` argument must be "
                        "a numpy array with the dtype `np.uint64`!")

        # determine the basinmap to use
        if basin_map is not None:
            self.h5file.require_group("events")
            if basin_map_name is None:
                # We have to determine the basin_map_name to use for this
                # mapped basin.
                for ii in range(10):  # basinmap0 to basinmap9
                    bm_cand = f"basinmap{ii}"
                    if bm_cand in self.h5file["events"]:
                        # There is a basin mapping defined in the file. Check
                        # whether it is identical to ours.
                        if np.all(self.h5file["events"][bm_cand] == basin_map):
                            # Great, we are done here.
                            basin_map_name = bm_cand
                            break
                        else:
                            # This mapping belongs to a different basin,
                            # try the next mapping.
                            continue
                    else:
                        # The mapping is not defined in the dataset, and we may
                        # write it to a new feature.
                        basin_map_name = bm_cand
                        self.store_feature(feat=basin_map_name, data=basin_map)
                        break
                else:
                    raise ValueError(
                        "You have exhausted the usage of mapped basins for "
                        "the current dataset. Please revise your analysis "
                        "pipeline.")
            else:
                if basin_map_name not in self.h5file["events"]:
                    # Write the explicit basin mapping into the file.
                    self.store_feature(feat=basin_map_name, data=basin_map)
                elif not np.all(
                        self.h5file["events"][basin_map_name] == basin_map):
                    # This is a sanity check that we have to perform.
                    raise ValueError(
                        f"The basin mapping feature {basin_map_name} you "
                        f"specified explicitly already exists in "
                        f"{self.h5file} and they do not match. I assume "
                        f"you are trying explicitly write to a basinmap that "
                        f"is already used elsewhere.")
        else:
            # Classic, simple case
            basin_map_name = "same"

        b_data = {
            "description": basin_descr,
            "format": basin_format,
            "name": basin_name,
            "type": basin_type,
            "features": None if basin_feats is None else sorted(basin_feats),
            "mapping": basin_map_name,
        }
        if basin_type == "file":
            flocs = []
            for pp in basin_locs:
                pp = pathlib.Path(pp)
                if verify:
                    flocs.append(str(pp.resolve()))
                    # Also store the relative path for user convenience.
                    # Don't use pathlib.Path.relative_to, because that
                    # is deprecated in Python 3.12.
                    # Also, just look in subdirectories which simplifies
                    # path resolution.
                    this_parent = str(self.path.parent) + os.sep
                    path_parent = str(pp.parent) + os.sep
                    if path_parent.startswith(this_parent):
                        flocs.append(str(pp).replace(this_parent, "", 1))
                else:
                    # We already did (or did not upon user request) verify
                    # the path. Just pass it on to the list.
                    flocs.append(str(pp))
            b_data["paths"] = flocs
        elif basin_type == "internal":
            b_data["paths"] = basin_locs
        elif basin_type == "remote":
            b_data["urls"] = [str(p) for p in basin_locs]
        else:
            raise ValueError(f"Unknown basin type '{basin_type}'")

        b_lines = json.dumps(b_data, indent=2).split("\n")
        basins = self.h5file.require_group("basins")
        key = hashobj(b_lines)
        if key not in basins:
            self.write_text(basins, key, b_lines)
        return key

    def store_feature(self, feat, data, shape=None):
        """Write feature data

        Parameters
        ----------
        feat: str
            feature name
        data: np.ndarray or list or dict
            feature data
        shape: tuple of int
            For non-scalar features, this is the shape of the
            feature for one event (e.g. `(90, 250)` for an "image".
            Usually, you do not have to specify this value, but you
            do need it in case of plugin features that don't have
            the "feature shape" set or in case of temporary features.
            If you don't specify it, then the shape is guessed based
            on the data you provide and a UserWarning will be issued.
        """
        if not dfn.feature_exists(feat):
            raise ValueError(f"Undefined feature '{feat}'!")

        events = self.h5file.require_group("events")

        # replace data?
        if feat in events and self.mode == "replace":
            if feat == "trace":
                for tr_name in data.keys():
                    if tr_name in events[feat]:
                        del events[feat][tr_name]
            else:
                del events[feat]

        if feat == "index":
            # By design, the index must be a simple enumeration.
            # We enforce that by not trusting the user. If you need
            # a different index, please take a look at the index_online
            # feature.
            nev = len(data)
            if "index" in events:
                nev0 = len(events["index"])
            else:
                nev0 = 0
            self.write_ndarray(group=events,
                               name="index",
                               data=np.arange(nev0 + 1, nev0 + nev + 1))
        elif dfn.scalar_feature_exists(feat):
            self.write_ndarray(group=events,
                               name=feat,
                               data=np.atleast_1d(data))
        elif feat == "contour":
            self.write_ragged(group=events, name=feat, data=data)
        elif feat in ["image", "image_bg", "mask", "qpi_oah", "qpi_oah_bg"]:
            self.write_image_grayscale(group=events,
                                       name=feat,
                                       data=data,
                                       is_boolean=(feat == "mask"))
        elif feat in ["qpi_amp", "qpi_pha"]:
            self.write_image_float32(group=events,
                                     name=feat,
                                     data=data)
        elif feat == "trace":
            for tr_name in data.keys():
                # verify trace names
                if tr_name not in dfn.FLUOR_TRACES:
                    raise ValueError(f"Unknown trace key: '{tr_name}'!")
                # write trace
                self.write_ndarray(group=events.require_group("trace"),
                                   name=tr_name,
                                   data=np.atleast_2d(data[tr_name])
                                   )
        else:
            if not shape:
                # OK, so we are dealing with a plugin feature or a temporary
                # feature here. Now, we don't know the exact shape of that
                # feature, but we give the user the option to advertise
                # the shape of the feature in the plugin.
                # First, try to obtain the shape from the PluginFeature
                # (if that exists).
                for pf in PlugInFeature.get_instances(feat):
                    if isinstance(pf, PlugInFeature):
                        shape = pf.plugin_feature_info.get("feature shape")
                        if shape is not None:
                            break  # This is good.
                else:
                    # Temporary features will have to live with this warning.
                    warnings.warn(
                        "There is no information about the shape of the "
                        + f"feature '{feat}'. I am going out on a limb "
                        + "for you and assume that you are storing "
                        + "multiple events at a time. If this works, "
                        + f"you could put the shape `{data[0].shape}` "
                        + 'in the `info["feature shapes"]` key of '
                        + "your plugin feature.")
                    shape = data.shape[1:]
            if shape == data.shape:
                data = data.reshape(1, *shape)
            elif shape == data.shape[1:]:
                pass
            else:
                raise ValueError(f"Bad shape for {feat}! Expeted {shape}, "
                                 + f"but got {data.shape[1:]}!")
            self.write_ndarray(group=events, name=feat, data=data)

    def store_log(self, name, lines):
        """Write log data

        Parameters
        ----------
        name: str
            name of the log entry
        lines: list of str or str
            the text lines of the log
        """
        log_group = self.h5file.require_group("logs")
        self.write_text(group=log_group, name=name, lines=lines)

    def store_metadata(self, meta):
        """Store RT-DC metadata

        Parameters
        ----------
        meta: dict-like
            The metadata to store. Each key depicts a metadata section
            name whose data is given as a dictionary, e.g.::

                meta = {"imaging": {"exposure time": 20,
                                    "flash duration": 2,
                                    ...
                                    },
                        "setup": {"channel width": 20,
                                  "chip region": "channel",
                                  ...
                                  },
                        ...
                        }

            Only section key names and key values therein registered
            in dclab are allowed and are converted to the pre-defined
            dtype. Only sections from the
            :const:`dclab.definitions.CFG_METADATA` dictionary are
            stored. If you have custom metadata, you can use the "user"
            section.
        """
        meta = copy.deepcopy(meta)
        # Ignore/remove tdms section
        meta.pop("fmt_tdms", None)
        # Check meta data
        for sec in meta:
            if sec == "user":
                # user-defined metadata are always written.
                # Any errors (incompatibilities with HDF5 attributes)
                # are the user's responsibility
                continue
            elif sec not in dfn.CFG_METADATA:
                # only allow writing of meta data that are not editable
                # by the user (not dclab.dfn.CFG_ANALYSIS)
                raise ValueError(
                    f"Meta data section not defined in dclab: {sec}")
            for ck in meta[sec]:
                if not dfn.config_key_exists(sec, ck):
                    raise ValueError(
                        f"Meta key not defined in dclab: {sec}:{ck}")

        # update version
        old_version = meta.get("setup", {}).get("software version", "")
        new_version = self.version_brand(
            old_version=old_version or None,
            write_attribute=False
        )
        meta.setdefault("setup", {})["software version"] = new_version

        # Write metadata
        for sec in meta:
            for ck in meta[sec]:
                idk = f"{sec}:{ck}"
                value = meta[sec][ck]
                if isinstance(value, bytes):
                    # We never store byte attribute values.
                    # In this case, `convfunc` should be `str` or `lcstr` or
                    # somesuch. But we don't test that, because no other
                    # datatype competes with str for bytes.
                    value = value.decode("utf-8")
                if sec == "user":
                    # store user-defined metadata as-is
                    self.h5file.attrs[idk] = value
                else:
                    # pipe the metadata through the hard-coded converter
                    # functions
                    convfunc = dfn.get_config_value_func(sec, ck)
                    self.h5file.attrs[idk] = convfunc(value)

    def store_table(self, name, cmp_array):
        """Store a compound array table

        Tables are semi-metadata. They may contain information collected
        during a measurement (but with a lower temporal resolution) or
        other tabular data relevant for a dataset. Tables have named
        columns. Therefore, they can be represented as a numy recarray,
        and they should be stored as such in an HDF5 file (compund dataset).

        Parameters
        ----------
        name: str
            Name of the table
        cmp_array: np.recarray, h5py.Dataset, or dict
            If a np.recarray or h5py.Dataset are provided, then they
            are written as-is to the file. If a dictionary is provided,
            then the dictionary is converted into a numpy recarray.
        """
        if isinstance(cmp_array, (np.recarray, h5py.Dataset)):
            # A table is a compound array (np.recarray). If we are here,
            # this means that the user passed an instance of np.recarray
            # or an instance h5py.Dataset (which we trust to be a proper
            # compound dataset at this point). No additional steps needed.
            pass
        elif isinstance(cmp_array, dict):
            # The user passed a dict which we now have to convert to a
            # compound dataset. We do this because we are user-convenient.
            # The user should not need to wade through these steps:
            columns = list(cmp_array.keys())
            # Everything should be floats in a table.
            ds_dt = np.dtype({'names': columns,
                              'formats': [np.float64] * len(columns)})
            # We trust the user to provide a dictionary with one-dimensional
            # lists or arrays of the same length.
            tabsize = len(cmp_array[columns[0]])
            tab_data = np.zeros((tabsize, len(columns)))
            for ii, tab in enumerate(columns):
                tab_data[:, ii] = cmp_array[tab]
            # Now create a new compound array (discarding the old dict)
            cmp_array = np.rec.array(tab_data, dtype=ds_dt)
        else:
            raise NotImplementedError(
                f"Cannot convert {type(cmp_array)} to table!")
        group = self.h5file.require_group("tables")
        tab = group.create_dataset(
            name,
            data=cmp_array,
            fletcher32=True,
            **self.compression_kwargs)
        # Also store metadata
        if hasattr(cmp_array, "attrs"):
            for key in cmp_array.attrs:
                tab.attrs[key] = cmp_array.attrs[key]

    def version_brand(self, old_version=None, write_attribute=True):
        """Perform version branding

        Append a " | dclab X.Y.Z" to the "setup:software version"
        attribute.

        Parameters
        ----------
        old_version: str or None
            By default, the version string is taken from the HDF5 file.
            If set to a string, then this version is used instead.
        write_attribute: bool
            If True (default), write the version string to the
            "setup:software version" attribute
        """
        if old_version is None:
            old_version = self.h5file.attrs.get("setup:software version", "")
        if isinstance(old_version, bytes):
            old_version = old_version.decode("utf-8")
        version_chain = [vv.strip() for vv in old_version.split("|")]
        version_chain = [vv for vv in version_chain if vv]
        cur_version = "dclab {}".format(version)

        if version_chain:
            if version_chain[-1] != cur_version:
                version_chain.append(cur_version)
        else:
            version_chain = [cur_version]
        new_version = " | ".join(version_chain)
        if write_attribute:
            self.h5file.attrs["setup:software version"] = new_version
        else:
            return new_version

    def write_image_float32(self, group, name, data):
        """Write 32bit floating point image array

        This function wraps :func:`RTDCWriter.write_ndarray`
        and adds image attributes to the HDF5 file so HDFView
        can display the images properly.

        Parameters
        ----------
        group: h5py.Group
            parent group
        name: str
            name of the dataset containing the text
        data: np.ndarray or list of np.ndarray
            image data
        """
        if isinstance(data, (list, tuple)):
            # images may be in lists
            data = np.atleast_2d(data)

        if len(data.shape) == 2:
            # put single event in 3D array
            data = data[np.newaxis]

        dset = self.write_ndarray(group=group, name=name, data=data,
                                  dtype=np.float32)

        # Create and Set image attributes:
        # HDFView recognizes this as a series of images.
        # Use np.bytes_ as per
        # https://docs.h5py.org/en/stable/strings.html#compatibility
        dset.attrs.create('CLASS', np.bytes_('IMAGE'))
        dset.attrs.create('IMAGE_VERSION', np.bytes_('1.2'))
        dset.attrs.create('IMAGE_SUBCLASS', np.bytes_('IMAGE_GRAYSCALE'))

    def write_image_grayscale(self, group, name, data, is_boolean):
        """Write grayscale image data to and HDF5 dataset

        This function wraps :func:`RTDCWriter.write_ndarray`
        and adds image attributes to the HDF5 file so HDFView
        can display the images properly.

        Parameters
        ----------
        group: h5py.Group
            parent group
        name: str
            name of the dataset containing the text
        data: np.ndarray or list of np.ndarray
            image data
        is_boolean: bool
            whether the input data is of boolean nature
            (e.g. mask data) - if so, data are converted to uint8
        """
        if isinstance(data, (list, tuple)):
            # images may be in lists
            data = np.atleast_2d(data)

        if len(data.shape) == 2:
            # put single event in 3D array
            data = data.reshape(1, data.shape[0], data.shape[1])

        if is_boolean:
            # convert binary (mask) data to uint8
            if data.__class__.__name__ == "H5MaskEvent":
                # (if we use `isinstance`, we get circular imports)
                # Be smart and directly write back the original data
                # (otherwise we would convert to bool and back to uint8).
                data = data.h5dataset
            elif data.dtype == bool:
                # Convert binary input mask data to uint8 with max range
                data = np.asarray(data, dtype=np.uint8) * 255

        dset = self.write_ndarray(group=group, name=name, data=data,
                                  dtype=np.uint8)

        # Create and Set image attributes:
        # HDFView recognizes this as a series of images.
        # Use np.bytes_ as per
        # https://docs.h5py.org/en/stable/strings.html#compatibility
        dset.attrs.create('CLASS', np.bytes_('IMAGE'))
        dset.attrs.create('IMAGE_VERSION', np.bytes_('1.2'))
        dset.attrs.create('IMAGE_SUBCLASS', np.bytes_('IMAGE_GRAYSCALE'))

    def write_ndarray(self, group, name, data, dtype=None):
        """Write n-dimensional array data to an HDF5 dataset

        It is assumed that the shape of the array data is correct,
        i.e. that the shape of `data` is
        (number_events, feat_shape_1, ..., feat_shape_n).

        Parameters
        ----------
        group: h5py.Group
            parent group
        name: str
            name of the dataset containing the text
        data: np.ndarray
            data
        dtype: dtype
            the dtype to use for storing the data
            (defaults to `data.dtype`)
        """
        if len(data) == 0:
            raise ValueError(f"Empty data object for '{name}'")

        if name not in group:
            chunks = self.get_best_nd_chunks(item_shape=data.shape[1:],
                                             item_dtype=data.dtype)
            maxshape = tuple([None] + list(data.shape)[1:])
            dset = group.create_dataset(
                name,
                shape=data.shape,
                dtype=dtype or data.dtype,
                maxshape=maxshape,
                chunks=chunks,
                fletcher32=True,
                **self.compression_kwargs)
            offset = 0
        else:
            dset = group[name]
            offset = dset.shape[0]
            dset.resize(offset + data.shape[0], axis=0)
        if len(data.shape) == 1:
            # store scalar data in one go
            dset[offset:] = data
            # store ufunc data for min/max
            for uname, ufunc in [("min", np.nanmin),
                                 ("max", np.nanmax)]:
                val_a = dset.attrs.get(uname, None)
                if val_a is not None:
                    val_b = ufunc(data)
                    val = ufunc([val_a, val_b])
                else:
                    val = ufunc(dset)
                dset.attrs[uname] = val
            # store ufunc data for mean (weighted with size)
            mean_a = dset.attrs.get("mean", None)
            if mean_a is not None:
                num_a = offset
                mean_b = np.nanmean(data)
                num_b = data.size
                mean = (mean_a * num_a + mean_b * num_b) / (num_a + num_b)
            else:
                mean = np.nanmean(dset)
            dset.attrs["mean"] = mean
        else:
            chunk_size = dset.chunks[0]
            # populate higher-dimensional data in chunks
            # (reduces file size, memory usage, and saves time)
            num_chunks = len(data) // chunk_size
            for ii in range(num_chunks):
                start = ii * chunk_size
                stop = start + chunk_size
                dset[offset+start:offset+stop] = data[start:stop]
            # write remainder (if applicable)
            num_remain = len(data) % chunk_size
            if num_remain:
                start_e = num_chunks * chunk_size
                stop_e = start_e + num_remain
                dset[offset+start_e:offset+stop_e] = data[start_e:stop_e]
        return dset

    def write_ragged(self, group, name, data):
        """Write ragged data (i.e. list of arrays of different lenghts)

        Ragged array data (e.g. contour data) are stored in
        a separate group and each entry becomes an HDF5 dataset.

        Parameters
        ----------
        group: h5py.Group
            parent group
        name: str
            name of the dataset containing the text
        data: list of np.ndarray or np.ndarray
            the data in a list
        """
        if isinstance(data, np.ndarray) and len(data.shape) == 2:
            # place single event in list
            data = [data]
        grp = group.require_group(name)
        # The following case is just a workaround for the very slow
        # `len(grp)` which makes things horrible if you are storing
        # contour data one-by-one. The only downside of this is that
        # we have to keep track of the length of the group. But I
        # think that is OK, since everything is very private here.
        # - Paul (2021-10-18)
        if grp not in self._group_sizes:
            self._group_sizes[grp] = len(grp)
        curid = self._group_sizes[grp]
        for ii, cc in enumerate(data):
            grp.create_dataset("{}".format(curid + ii),
                               data=cc,
                               fletcher32=True,
                               chunks=cc.shape,
                               **self.compression_kwargs)
            self._group_sizes[grp] += 1

    def write_text(self, group, name, lines):
        """Write text to an HDF5 dataset

        Text data are written as a fixed-length string dataset.

        Parameters
        ----------
        group: h5py.Group
            parent group
        name: str
            name of the dataset containing the text
        lines: list of str or str
            the text, line by line
        """
        # replace text?
        if name in group and self.mode == "replace":
            del group[name]

        # handle strings
        if isinstance(lines, (str, bytes)):
            lines = [lines]

        lnum = len(lines)
        # Determine the maximum line length and use fixed-length strings,
        # because compression and fletcher32 filters won't work with
        # variable length strings.
        # https://github.com/h5py/h5py/issues/1948
        # 100 is the recommended maximum and the default, because if
        # `mode` is e.g. "append", then this line may not be the longest.
        max_length = 100
        lines_as_bytes = []
        for line in lines:
            # convert lines to bytes
            if not isinstance(line, bytes):
                lbytes = line.encode("UTF-8")
            else:
                lbytes = line
            max_length = max(max_length, len(lbytes))
            lines_as_bytes.append(lbytes)

        if name not in group:
            # Create the dataset
            txt_dset = group.create_dataset(
                name,
                shape=(lnum,),
                dtype=f"S{max_length}",
                maxshape=(None,),
                chunks=True,
                fletcher32=True,
                **self.compression_kwargs)
            line_offset = 0
        else:
            # TODO: test whether fixed length is long enough!
            # Resize the dataset
            txt_dset = group[name]
            line_offset = txt_dset.shape[0]
            txt_dset.resize(line_offset + lnum, axis=0)

        # Write the text data line-by-line
        for ii, lbytes in enumerate(lines_as_bytes):
            txt_dset[line_offset + ii] = lbytes
