#!/usr/bin/python
# -*- coding: utf-8 -*-
"""RT-DC hdf5 format"""
from __future__ import division, print_function, unicode_literals

from distutils.version import LooseVersion
import os
import warnings

import h5py

from dclab import definitions as dfn
from .config import Configuration
from .core import RTDCBase
from .util import hashobj, hashfile


#: rtdc files exported with dclab prior to this version are not supported
MIN_DCLAB_EXPORT_VERSION = "0.3.3.dev11"


class OldFormatNotSupportedError(BaseException):
    pass


class UnknownKeyWarning(UserWarning):
    pass


class H5Events(object):
    def __init__(self, h5path):
        self.path = h5path
        self._h5 = h5py.File(h5path, mode="r")

    def __contains__(self, key):
        return key in self.keys()

    def __getitem__(self, key):
        # user-level checking is done in core.py
        assert key in dfn.feature_names + ["contour", "image", "trace"]
        data = self._h5["events"][key]
        if key in dfn.feature_names:
            return data.value
        elif key in ["image", "trace"]:
            return data
        elif key == "contour":
            return H5ContourEvent(data)

    def keys(self):
        return sorted(list(self._h5["events"].keys()))


class H5ContourEvent(object):
    def __init__(self, h5group):
        self.h5group = h5group
        self.identifier = h5group["0"].value

    def __getitem__(self, key):
        return self.h5group[str(key)].value

    def __len__(self):
        return len(self.h5group)


class RTDC_HDF5(RTDCBase):
    def __init__(self, h5path, *args, **kwargs):
        """HDF5 file format for RT-DC measurements

        Parameters
        ----------
        h5path: str
            Path to a '.tdms' measurement file.
        *args, **kwargs:
            (Keyword) arguments for `RTDCBase`
        """
        super(RTDC_HDF5, self).__init__(*args, **kwargs)

        self._hash = None
        self.path = h5path

        # Setup events
        self._events = H5Events(h5path)

        # Parse configuration
        self.config = RTDC_HDF5.parse_config(h5path)

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

        self.title = self.config["experiment"]["sample"]

        # Set up filtering
        self._init_filters()

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
            if pname not in dfn.config_funcs[section]:
                msg = "Unknown key {} in section [{}]!".format(key, section)
                warnings.warn(msg, UnknownKeyWarning)
            else:
                typ = dfn.config_funcs[section][pname]
                config[section][pname] = typ(h5attrs[key])
        return config

    @property
    def hash(self):
        """Hash value based on file name and content"""
        if self._hash is None:
            tohash = [os.path.basename(self.path)]
            # Hash a maximum of ~1MB of the hdf5 file
            tohash.append(hashfile(self.path, blocksize=65536, count=20))
            self._hash = hashobj(tohash)
        return self._hash
