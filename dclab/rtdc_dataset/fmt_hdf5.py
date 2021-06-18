"""RT-DC hdf5 format"""
from distutils.version import LooseVersion
import numbers
import pathlib

import h5py
import numpy as np

from .. import definitions as dfn
from ..util import hashobj, hashfile

from .config import Configuration
from .core import RTDCBase


#: rtdc files exported with dclab prior to this version are not supported
MIN_DCLAB_EXPORT_VERSION = "0.3.3.dev2"

#: dictionary of defective features, defined by HDF5 attributes;
#: if a value matches the given HDF5 attribute, the feature is
#: considered defective
DEFECTIVE_FEATURES = {
    # feature: [HDF5_attribute, matching_value]
    # In Shape-In 2.0.6, there was a wrong variable cast for the
    # feature "aspect".
    "aspect": [["setup:software version", "ShapeIn 2.0.6"],
               ["setup:software version", "ShapeIn 2.0.7"],
               ]
}


class OldFormatNotSupportedError(BaseException):
    pass


class UnknownKeyWarning(UserWarning):
    pass


class H5ContourEvent(object):
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

    def __len__(self):
        return len(self.h5group)


class H5Events(object):
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
        if key in ["image", "image_bg", "trace"]:
            return data
        elif key == "contour":
            return H5ContourEvent(data)
        elif key == "mask":
            return H5MaskEvent(data)
        else:
            return data[:]

    def __iter__(self):
        # dict-like behavior
        for key in self.keys():
            yield key

    def _is_defective_feature(self, key):
        """Whether or not the stored feature is defective"""
        defective = False
        if key in self._features:
            # feature exists in the HDF5 file
            if key in DEFECTIVE_FEATURES:
                # workaround machinery for sorting out defective features
                for attr, value in DEFECTIVE_FEATURES[key]:
                    if attr in self._h5.attrs:
                        defective = self._h5.attrs[attr] == value
                    if defective:
                        break
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


class H5Logs(object):
    def __init__(self, h5):
        self._h5 = h5

    def __getitem__(self, key):
        if "logs" in self._h5:
            log = list(self._h5["logs"][key])
            if isinstance(log[0], bytes):
                log = [li.decode("utf") for li in log]
        else:
            raise KeyError("No logs in {}!".format(self._h5.file.filename))
        return log

    def __iter__(self):
        # dict-like behavior
        for key in self.keys():
            yield key

    def __len__(self):
        return len(self.keys())

    def keys(self):
        if "logs" in self._h5:
            names = sorted(self._h5["logs"].keys())
        else:
            names = []
        return names


class H5MaskEvent(object):
    """Cast uint8 masks to boolean"""

    def __init__(self, h5group):
        self.h5group = h5group
        # identifier required because "mask" is used for computation
        # of ancillary feature "contour".
        self.identifier = str(self.h5group.parent.parent.file)

    def __getitem__(self, idx):
        return np.asarray(self.h5group[idx], dtype=bool)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self):
        return len(self.h5group)

    @property
    def attrs(self):
        return self.h5group.attrs

    @property
    def shape(self):
        return self.h5group.shape


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
            rtdc_ver = LooseVersion(rtdc_soft.split(" ")[1])
            if rtdc_ver < LooseVersion(MIN_DCLAB_EXPORT_VERSION):
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
        """Parse the RT-DC configuration of an hdf5 file"""
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
            if section == "user":
                # the type of user configuration parameters are defined
                # by the HDF5 attribute type
                config[section][pname] = h5attrs[key]
            elif pname not in dfn.config_funcs[section]:
                # Add the value as a string (this will issue
                # a UnknownConfigurationKeyWarning in config.py)
                config[section][pname] = h5attrs[key]
            else:
                typ = dfn.config_funcs[section][pname]
                config[section][pname] = typ(h5attrs[key])

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
