"""RT-DC hdf5 format"""
from __future__ import annotations

import io
import json
import pathlib
from typing import Any, BinaryIO, Dict
import warnings

import h5py

from ...external.packaging import parse as parse_version
from ...util import hashobj, hashfile

from ..config import Configuration
from ..core import RTDCBase

from . import events
from . import logs
from . import tables

#: rtdc files exported with dclab prior to this version are not supported
MIN_DCLAB_EXPORT_VERSION = "0.3.3.dev2"


class OldFormatNotSupportedError(BaseException):
    pass


class UnknownKeyWarning(UserWarning):
    pass


class RTDC_HDF5(RTDCBase):
    def __init__(self,
                 h5path: str | pathlib.Path | BinaryIO | io.IOBase,
                 h5kwargs: Dict[str, Any] = None,
                 *args,
                 **kwargs):
        """HDF5 file format for RT-DC measurements

        Parameters
        ----------
        h5path: str or pathlib.Path or file-like object
            Path to an '.rtdc' measurement file or a file-like object
        h5kwargs: dict
            Additional keyword arguments given to :class:`h5py.File`
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

        # Any subclass from RTDC_HDF5 is probably a remote-type and should
        # not be able to access local basins. If you do not agree, please
        # enable this in the definition of the subclass.
        self._local_basins_allowed = True if self.format == "hdf5" else False

        if isinstance(h5path, (str, pathlib.Path)):
            h5path = pathlib.Path(h5path)
        else:
            h5path = h5path

        self._hash = None
        self.path = h5path

        # Increase the read cache (which defaults to 1MiB), since
        # normally we have around 2.5MiB image chunks.
        if h5kwargs is None:
            h5kwargs = {}
        h5kwargs.setdefault("rdcc_nbytes", 10 * 1024 ** 2)
        h5kwargs.setdefault("rdcc_w0", 0)

        self.h5kwargs = h5kwargs
        self.h5file = h5py.File(h5path, **h5kwargs)

        self._events = events.H5Events(self.h5file)

        # Parse configuration
        self.config = RTDC_HDF5.parse_config(self.h5file)

        # Override logs property with HDF5 data
        self.logs = logs.H5Logs(self.h5file)

        # Override the tables property with HDF5 data
        self.tables = tables.H5Tables(self.h5file)

        # check version
        rtdc_soft = self.config["setup"].get("software version", "unknown")
        if rtdc_soft.startswith("dclab "):
            rtdc_ver = parse_version(rtdc_soft.split(" ")[1])
            if rtdc_ver < parse_version(MIN_DCLAB_EXPORT_VERSION):
                msg = "The file {} was created ".format(self.path) \
                      + "with dclab {} which is ".format(rtdc_ver) \
                      + "not supported anymore! Please rerun " \
                      + "dclab-tdms2rtdc / export the data again."
                raise OldFormatNotSupportedError(msg)

        self.title = "{} - M{}".format(
            self.config["experiment"].get("sample", "undefined sample"),
            self.config["experiment"].get("run index", "0"))

    def close(self):
        """Close the underlying HDF5 file"""
        super(RTDC_HDF5, self).close()
        self.h5file.close()

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
        """Parse the RT-DC configuration of an HDF5 file

        `h5path` may be a h5py.File object or an actual path
        """
        if not isinstance(h5path, h5py.File):
            with h5py.File(h5path, mode="r") as fh5:
                h5attrs = dict(fh5.attrs)
        else:
            h5attrs = dict(h5path.attrs)

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

    def basins_get_dicts(self):
        """Return list of dicts for all basins defined in `self.h5file`"""
        basins = []
        # Do not sort anything here, sorting is done in `RTDCBase`.
        for bk in self.h5file.get("basins", []):
            bdat = list(self.h5file["basins"][bk])
            if isinstance(bdat[0], bytes):
                bdat = [bi.decode("utf") for bi in bdat]
            bdict = json.loads(" ".join(bdat))
            bdict["key"] = bk
            basins.append(bdict)
        return basins
