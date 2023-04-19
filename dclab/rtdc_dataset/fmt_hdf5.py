"""RT-DC hdf5 format"""
import functools
import numbers
import pathlib
import warnings

import h5py
import numpy as np

from .. import definitions as dfn
from ..external.packaging import parse as parse_version
from ..util import hashobj, hashfile

from .config import Configuration
from .core import RTDCBase


class OldFormatNotSupportedError(BaseException):
    pass


class UnknownKeyWarning(UserWarning):
    pass


class H5ContourEvent:
    def __init__(self, h5group):
        self.h5group = h5group
        # for hashing in util.obj2bytes
        self.identifier = (h5group.file.filename, h5group["0"].name)

    def __getitem__(self, key):
        if not isinstance(key, numbers.Integral):
            # slicing!
            indices = np.arange(len(self))[key]
            output = []
            # populate the output list
            for evid in indices:
                output.append(self.h5group[str(evid)][:])
            return output
        elif key < 0:
            return self.__getitem__(key + len(self))
        else:
            return self.h5group[str(key)][:]

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    @functools.lru_cache()
    def __len__(self):
        # computing the length of an H5Group is slow
        return len(self.h5group)

    @property
    def shape(self):
        return len(self), np.nan, 2


class H5Events:
    def __init__(self, h5):
        self.h5file = h5
        # According to https://github.com/h5py/h5py/issues/1960, we always
        # have to keep a reference to the HDF5 dataset, otherwise it will
        # be garbage-collected immediately. In addition to caching the HDF5
        # datasets, we cache the wrapping classes in the `self._cached_events`
        # dictionary.
        self._cached_events = {}
        self._features = sorted(self.h5file["events"].keys())
        # make sure that "trace" is not empty
        if ("trace" in self._features
                and len(self.h5file["events"]["trace"]) == 0):
            self._features.remove("trace")

    def __contains__(self, key):
        return key in self.keys()

    def __getitem__(self, key):
        if key not in self._cached_events:
            # user-level checking is done in core.py
            assert dfn.feature_exists(key), f"Feature '{key}' does not exist!"
            data = self.h5file["events"][key]
            if key == "contour":
                fdata = H5ContourEvent(data)
            elif key == "mask":
                fdata = H5MaskEvent(data)
            elif key == "trace":
                fdata = H5TraceEvent(data)
            elif data.ndim == 1:
                fdata = H5ScalarEvent(data)
            else:
                # for features like "image", "image_bg" and other non-scalar
                # ancillary features
                fdata = data
            self._cached_events[key] = fdata
        return self._cached_events[key]

    def __iter__(self):
        # dict-like behavior
        for key in self.keys():
            yield key

    @functools.lru_cache()
    def _is_defective_feature(self, feat):
        """Whether the stored feature is defective"""
        defective = False
        if feat in DEFECTIVE_FEATURES and feat in self._features:
            # feature exists in the HDF5 file
            # workaround machinery for sorting out defective features
            defective = DEFECTIVE_FEATURES[feat](self.h5file)
        return defective

    def keys(self):
        """Returns list of valid features

        Checks for
        - defective features: whether the data in the HDF5 file is invalid
        - existing feature names: dynamic, depending on e.g. plugin features
        """
        features = []
        for key in self._features:
            # check for defective features
            if dfn.feature_exists(key) and not self._is_defective_feature(key):
                features.append(key)
        return features


class H5Logs:
    def __init__(self, h5):
        self.h5file = h5

    def __getitem__(self, key):
        if key in self.keys():
            log = list(self.h5file["logs"][key])
            if isinstance(log[0], bytes):
                log = [li.decode("utf") for li in log]
        else:
            raise KeyError(
                f"File {self.h5file.file.filename} does not have the log "
                f"'{key}'. Available logs are {self.keys()}.")
        return log

    def __iter__(self):
        # dict-like behavior
        for key in self.keys():
            yield key

    def __len__(self):
        return len(self.keys())

    @functools.lru_cache()
    def keys(self):
        names = []
        if "logs" in self.h5file:
            for key in self.h5file["logs"]:
                if self.h5file["logs"][key].size:
                    names.append(key)
        return names


class H5MaskEvent:
    """Cast uint8 masks to boolean"""

    def __init__(self, h5dataset):
        self.h5dataset = h5dataset
        # identifier required because "mask" is used for computation
        # of ancillary feature "contour".
        self.identifier = (self.h5dataset.file.filename, self.h5dataset.name)
        self.dtype = np.dtype(bool)

    def __getitem__(self, idx):
        return np.asarray(self.h5dataset[idx], dtype=bool)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self):
        return len(self.h5dataset)

    @property
    def attrs(self):
        return self.h5dataset.attrs

    @property
    def shape(self):
        return self.h5dataset.shape


class H5ScalarEvent(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, h5ds):
        """Lazy access to a scalar feature with cache"""
        self.h5ds = h5ds
        # for hashing in util.obj2bytes
        self.identifier = (self.h5ds.file.filename, self.h5ds.name)
        self._array = None
        self.ndim = 1  # matplotlib might expect this from an array
        # attrs
        self._ufunc_attrs = dict(self.h5ds.attrs)

    def __array__(self, dtype=None):
        if self._array is None:
            self._array = np.asarray(self.h5ds, dtype=dtype)
        return self._array

    def __getitem__(self, idx):
        return self.__array__()[idx]

    def __len__(self):
        return len(self.h5ds)

    def _fetch_ufunc_attr(self, uname, ufunc):
        """A wrapper for calling functions on the scalar feature data

        The ideas are:

        1. If there is a ufunc (max/mean/min) value stored in the dataset
           attributes, then use this one.
        2. If the ufunc is computed, it is cached permanently in
           self._ufunc_attrs
        """
        val = self._ufunc_attrs.get(uname, None)
        if val is None:
            val = ufunc(self.__array__())
            self._ufunc_attrs[uname] = val
        return val

    def max(self, *args, **kwargs):
        return self._fetch_ufunc_attr("max", np.nanmax)

    def mean(self, *args, **kwargs):
        return self._fetch_ufunc_attr("mean", np.nanmean)

    def min(self, *args, **kwargs):
        return self._fetch_ufunc_attr("min", np.nanmin)

    @property
    def shape(self):
        return self.h5ds.shape


class H5Tables:
    def __init__(self, h5):
        self.h5file = h5

    def __getitem__(self, key):
        if key in self.keys():
            tab = self.h5file["tables"][key]
        else:
            raise KeyError(f"Table '{key}' not found or empty "
                           f"in {self.h5file.file.filename}!")
        return tab

    def __iter__(self):
        # dict-like behavior
        for key in self.keys():
            yield key

    def __len__(self):
        return len(self.keys())

    @functools.lru_cache()
    def keys(self):
        names = []
        if "tables" in self.h5file:
            for key in self.h5file["tables"]:
                if self.h5file["tables"][key].size:
                    names.append(key)
        return names


class H5TraceEvent:
    def __init__(self, h5group):
        self.h5group = h5group

    def __getitem__(self, idx):
        return self.h5group[idx]

    def __contains__(self, item):
        return item in self.h5group

    def __len__(self):
        return len(self.h5group)

    def __iter__(self):
        for key in sorted(self.h5group.keys()):
            yield key

    def keys(self):
        return self.h5group.keys()

    @property
    def shape(self):
        atrace = list(self.h5group.keys())[0]
        return tuple([len(self.h5group)] + list(self.h5group[atrace].shape))


class RTDC_HDF5(RTDCBase):
    def __init__(self, h5path, *args, **kwargs):
        """HDF5 file format for RT-DC measurements

        Parameters
        ----------
        h5path: str or pathlib.Path
            Path to a '.tdms' measurement file.
        *args:
            Arguments for `RTDCBase`
        **kwargs:
            Keyword arguments for `RTDCBase`

        Attributes
        ----------
        path: pathlib.Path
            Path to the experimental HDF5 (.rtdc) file
        """
        super(RTDC_HDF5, self).__init__(*args, **kwargs)

        h5path = pathlib.Path(h5path)
        self._hash = None
        self.path = h5path

        # Setup events
        self.h5file = h5py.File(
            h5path,
            mode="r",
            # Increase the read cache (which defaults to 1MiB), since
            # normally we have around 2.5MiB image chunks.
            rdcc_nbytes=10*1024**2,
            rdcc_w0=0,
            )
        self._events = H5Events(self.h5file)

        # Parse configuration
        self.config = RTDC_HDF5.parse_config(h5path)

        # Override logs property with HDF5 data
        self.logs = H5Logs(self.h5file)

        # Override the tables property with HDF5 data
        self.tables = H5Tables(self.h5file)

        # check version
        rtdc_soft = self.config["setup"]["software version"]
        if rtdc_soft.startswith("dclab "):
            rtdc_ver = parse_version(rtdc_soft.split(" ")[1])
            if rtdc_ver < parse_version(MIN_DCLAB_EXPORT_VERSION):
                msg = "The file {} was created ".format(self.path) \
                      + "with dclab {} which is ".format(rtdc_ver) \
                      + "not supported anymore! Please rerun " \
                      + "dclab-tdms2rtdc / export the data again."
                raise OldFormatNotSupportedError(msg)

        self.title = "{} - M{}".format(self.config["experiment"]["sample"],
                                       self.config["experiment"]["run index"])

        # Set up filtering
        self._init_filters()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        # close the HDF5 file
        self.h5file.close()

    @functools.lru_cache()
    def __len__(self):
        ec = self.h5file.get("experiment:event count")
        if ec is not None:
            return ec
        else:
            return super(RTDC_HDF5, self).__len__()

    @property
    def _h5(self):
        warnings.warn("Access to the underlying HDF5 file is now public. "
                      "Please use the `h5file` attribute instead of `_h5`!",
                      DeprecationWarning)
        return self.h5file

    @staticmethod
    def can_open(h5path):
        """Check whether a given file is in the .rtdc file format"""
        h5path = pathlib.Path(h5path)
        if h5path.suffix == ".rtdc":
            return True
        else:
            # we don't know the extension; check for the "events" group
            canopen = False
            try:
                # This is a workaround for Python2 where h5py cannot handle
                # unicode file names.
                with h5path.open("rb") as fd:
                    h5 = h5py.File(fd, "r")
                    if "events" in h5:
                        canopen = True
            except IOError:
                # not an HDF5 file
                pass
            return canopen

    @staticmethod
    def parse_config(h5path):
        """Parse the RT-DC configuration of an HDF5 file"""
        with h5py.File(h5path, mode="r") as fh5:
            h5attrs = dict(fh5.attrs)

        # Convert byte strings to unicode strings
        # https://github.com/h5py/h5py/issues/379
        for key in h5attrs:
            if isinstance(h5attrs[key], bytes):
                h5attrs[key] = h5attrs[key].decode("utf-8")

        config = Configuration()
        for key in h5attrs:
            section, pname = key.split(":")
            config[section][pname] = h5attrs[key]
        return config

    @property
    def hash(self):
        """Hash value based on file name and content"""
        if self._hash is None:
            tohash = [self.path.name,
                      # Hash a maximum of ~1MB of the hdf5 file
                      hashfile(self.path, blocksize=65536, count=20)]
            self._hash = hashobj(tohash)
        return self._hash


def get_software_version_from_h5(h5):
    software_version = h5.attrs.get("setup:software version", "")
    if isinstance(software_version, bytes):
        software_version = software_version.decode("utf-8")
    return software_version


def is_defective_feature_aspect(h5):
    """In Shape-In 2.0.6, there was a wrong variable cast"""
    software_version = get_software_version_from_h5(h5)
    return software_version in ["ShapeIn 2.0.6", "ShapeIn 2.0.7"]


def is_defective_feature_time(h5):
    """Shape-In stores the "time" feature as a low-precision float32

    This makes time resolution for large measurements useless,
    because times are only resolved with four digits after the
    decimal point. Here, we first check whether the "frame" feature
    and the [imaging]:"frame rate" configuration are set. If so,
    then we can compute "time" as an ancillary feature which will
    be more accurate than its float32 version.
    """
    # This is a necessary requirement. If we cannot compute the
    # ancillary feature, then we cannot ignore (even inaccurate) information.
    has_ancil = "frame" in h5["events"] and h5.attrs.get("imaging:frame rate",
                                                         0) != 0
    if not has_ancil:
        return False

    # If we have a 32 bit dataset, then things are pretty clear.
    is_32float = h5["events/time"].dtype.char[-1] == "f"
    if is_32float:
        return True

    # Consider the software
    software_version = get_software_version_from_h5(h5)

    # Only Shape-In stores false data, so we can ignore other recording
    # software.
    is_shapein = software_version.count("ShapeIn")
    if not is_shapein:
        return False

    # The tricky part: dclab might have analyzed the dataset recorded by
    # Shape-In, e.g. in a compression step. Since dclab appends its version
    # string to the software_version, we just have to parse that and make
    # sure that it is above 0.47.6.
    last_version = software_version.split("|")[-1].strip()
    if last_version.startswith("dclab"):
        dclab_version = last_version.split()[1]
        if parse_version(dclab_version) < parse_version("0.47.6"):
            # written with an older version of dclab
            return True

    # We covered all cases:
    # - ancillary information are available
    # - it's not a float32 dataset
    # - we excluded all non-Shape-In recording software
    # - it was not written with an older version of dclab
    return False


def is_defective_feature_volume(h5):
    """dclab computed volume wrong up until version 0.36.1"""
    # first check if the scripted fix was applied
    if "dclab_issue_141" in list(h5.get("logs", {}).keys()):
        return False
    # if that does not apply, check the software version
    software_version = get_software_version_from_h5(h5)
    if software_version:
        last_version = software_version.split("|")[-1].strip()
        if last_version.startswith("dclab"):
            dclab_version = last_version.split()[1]
            if parse_version(dclab_version) < parse_version("0.37.0"):
                return True
    return False


def is_defective_feature_inert_ratio(h5):
    """For long channels, there was an integer overflow until 0.48.1

    The problem here is that not only the channel length, but also
    the length of the contour play a role. All point coordinates of
    the contour are summed up and multiplied with one another which
    leads to integer overflows when computing mu20.

    Thus, this test is only a best guess, but still quite fast.

    See also https://github.com/DC-analysis/dclab/issues/212
    """
    # determine whether the image width is larger than 500px
    # If this file was written with dclab, then we always have the ROI size,
    # so we don't have to check the actual image.
    width_large = h5.attrs.get("imaging:roi size x", 0) > 500

    if width_large:
        # determine whether the software version was outdated
        software_version = get_software_version_from_h5(h5)
        if software_version:
            last_version = software_version.split("|")[-1].strip()
            if last_version.startswith("dclab"):
                dclab_version = last_version.split()[1]
                # The fix was implemented in 0.48.2, but this method here
                # was only implemented in 0.48.3, so we might have leaked
                # old data into new files.
                if parse_version(dclab_version) < parse_version("0.48.3"):
                    return True
    return False


#: rtdc files exported with dclab prior to this version are not supported
MIN_DCLAB_EXPORT_VERSION = "0.3.3.dev2"

#: dictionary of defective features, defined by HDF5 attributes;
#: if a value matches the given HDF5 attribute, the feature is
#: considered defective
DEFECTIVE_FEATURES = {
    # feature: [HDF5_attribute, matching_value]
    "aspect": is_defective_feature_aspect,
    "inert_ratio_cvx": is_defective_feature_inert_ratio,
    "inert_ratio_prnc": is_defective_feature_inert_ratio,
    "inert_ratio_raw": is_defective_feature_inert_ratio,
    "tilt": is_defective_feature_inert_ratio,
    "time": is_defective_feature_time,
    "volume": is_defective_feature_volume,
}
