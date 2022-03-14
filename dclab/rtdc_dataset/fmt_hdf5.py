"""RT-DC hdf5 format"""
import functools
import numbers
import pathlib

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
        self.identifier = h5group["0"][:]

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
        self._h5 = h5
        self._features = sorted(self._h5["events"].keys())
        # make sure that "trace" is not empty
        if "trace" in self._features and len(self._h5["events"]["trace"]) == 0:
            self._features.remove("trace")

    def __contains__(self, key):
        return key in self.keys()

    def __getitem__(self, key):
        # user-level checking is done in core.py
        assert dfn.feature_exists(key), "Feature '{}' not valid!".format(key)
        data = self._h5["events"][key]
        if key == "contour":
            return H5ContourEvent(data)
        elif key == "mask":
            return H5MaskEvent(data)
        elif key == "trace":
            return H5TraceEvent(data)
        elif data.ndim == 1:
            return data[:]
        else:
            # for features like "image", "image_bg" and other non-scalar
            # ancillary features
            return data

    def __iter__(self):
        # dict-like behavior
        for key in self.keys():
            yield key

    def _is_defective_feature(self, feat):
        """Whether or not the stored feature is defective"""
        defective = False
        if feat in DEFECTIVE_FEATURES and feat in self._features:
            # feature exists in the HDF5 file
            # workaround machinery for sorting out defective features
            defective = DEFECTIVE_FEATURES[feat](self._h5)
        return defective

    def keys(self):
        """Returns list of valid features

        Checks for
        - defective features
        - existing feature names
        """
        features = []
        for key in self._features:
            # check for defective features
            if dfn.feature_exists(key) and not self._is_defective_feature(key):
                features.append(key)
        return features


class H5Logs:
    def __init__(self, h5):
        self._h5 = h5

    def __getitem__(self, key):
        if key in self.keys():
            log = list(self._h5["logs"][key])
            if isinstance(log[0], bytes):
                log = [li.decode("utf") for li in log]
        else:
            raise KeyError(
                f"Log '{key}' not found or empty in {self._h5.file.filename}!")
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
        if "logs" in self._h5:
            for key in self._h5["logs"]:
                if self._h5["logs"][key].size:
                    names.append(key)
        return names


class H5MaskEvent:
    """Cast uint8 masks to boolean"""

    def __init__(self, h5dataset):
        self.h5dataset = h5dataset
        # identifier required because "mask" is used for computation
        # of ancillary feature "contour".
        self.identifier = str(self.h5dataset.parent.parent.file)

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
        self._h5 = h5py.File(h5path, mode="r")
        self._events = H5Events(self._h5)

        # Parse configuration
        self.config = RTDC_HDF5.parse_config(h5path)

        # Override logs property with HDF5 data
        self.logs = H5Logs(self._h5)

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
        self._h5.close()

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


def is_defective_feature_aspect(h5):
    """In Shape-In 2.0.6, there was a wrong variable cast"""
    software_version = h5.attrs["setup:software version"]
    if isinstance(software_version, bytes):
        software_version = software_version.decode("utf-8")
    return software_version in ["ShapeIn 2.0.6", "ShapeIn 2.0.7"]


def is_defective_feature_volume(h5):
    """dclab computed volume wrong up until version 0.36.1"""
    # first check if the scripted fix was applied
    if "dclab_issue_141" in list(h5.get("logs", {}).keys()):
        return False
    # if that does not apply, check the software version
    software_version = h5.attrs["setup:software version"]
    if isinstance(software_version, bytes):
        software_version = software_version.decode("utf-8")
    if software_version:
        last_version = software_version.split("|")[-1].strip()
        if last_version.startswith("dclab"):
            dclab_version = last_version.split()[1]
            if parse_version(dclab_version) < parse_version("0.37.0"):
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
    "volume": is_defective_feature_volume,
}
