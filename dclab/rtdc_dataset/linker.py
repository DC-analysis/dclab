"""Tools for linking HDF5 datasets across files"""
from __future__ import annotations

import io
import pathlib
from typing import BinaryIO, Literal

import h5py


class ExternalDataForbiddenError(BaseException):
    """Raised when a dataset contains external data

    External data are a security risk, because they could be
    used to access data that are not supposed to be accessed.
    This is especially critical when the data are accessed within
    a web server process (e.g. in DCOR).
    """
    pass


def assert_no_external(h5):
    """Raise ExternalDataForbiddenError if `h5` refers to external data"""
    has_ext, path_ext = check_external(h5)
    if has_ext:
        raise ExternalDataForbiddenError(
            f"Dataset {h5.file.filename} contains external data, but these "
            f"are not permitted for security reasons ({path_ext})!")


def check_external(h5):
    """Check recursively, whether an h5py object contains external data

    External data includes binary data in external files, virtual
    datasets, and external links.

    Returns a tuple of either

    - `(True, path_ext)` if the object contains external data
    - `(False, None)` if this is not the case

    where `path_ext` is the path to the group or dataset in `h5`.

    .. versionadded:: 0.51.0

    """
    for key in h5:
        obj = h5[key]
        if (obj.file != h5.file  # not in same file
                or (isinstance(obj, h5py.Dataset)
                    and (obj.is_virtual  # virtual dataset
                         or obj.external))):  # external dataset
            # These are external data
            return True, f"{h5.name}/{key}".replace("//", "/")
        elif isinstance(obj, h5py.Group):
            # Perform recursive check for external data
            has_ext, path_ext = check_external(obj)
            if has_ext:
                return True, path_ext
    else:
        return False, None


def combine_h5files(
        paths: list,
        external: Literal["follow", "raise"] = "follow"
        ) -> BinaryIO:
    """Create an in-memory file that combines multiple .rtdc files

    The .rtdc files must have the same number of events. The in-memory
    file is populated with the "events" data from `paths` according to
    the order that `paths` are given in. Metadata, including logs, basins,
    and tables are only taken from the first path.

    .. versionadded:: 0.51.0

    Parameters
    ----------
    paths: list of str or pathlib.Path
        Paths of the input .rtdc files. The first input file is always
        used as a source for the metadata. The other files only complement
        the features.
    external: str
        Defines how external (links, binary, virtual) data in `paths`
        should be handled. The default is to "follow" external datasets or
        links to external data. In a zero-trust context, you can set this
        to "raise" which will cause an :class:`.ExternalDataForbiddenError`
        exception when external data are encountered.

    Returns
    -------
    fd: BinaryIO
        seekable, file-like object representing an HDF5 file opened in
        binary mode; This can be passed to `:class:h5py.File`
    """
    fd = io.BytesIO()
    with h5py.File(fd, "w", libver="latest") as hv:
        for ii, pp in enumerate(paths):
            pp = pathlib.Path(pp).resolve()
            with h5py.File(pp, libver="latest") as h5:
                if external == "raise":
                    # Check for external data
                    assert_no_external(h5)
                if ii == 0:
                    # Only write attributes once.
                    # Interestingly, writing the attributes takes
                    # the most time. Maybe there is some shortcut
                    # that can be taken (since e.g. we know we don't have to
                    # check for existing attributes).
                    # https://github.com/h5py/h5py/blob/master/
                    # h5py/_hl/attrs.py
                    hv.attrs.update(h5.attrs)
                    # Also, write basins/logs/tables/... (anything that is
                    # not events) only once.
                    for group in h5:
                        if group != "events":
                            hv[group] = h5py.ExternalLink(str(pp), group)
                # Append features
                hve = hv.require_group("events")
                for feat in h5["events"]:
                    if feat not in hve:
                        hve[feat] = h5py.ExternalLink(str(pp),
                                                      f"/events/{feat}")
    return fd
