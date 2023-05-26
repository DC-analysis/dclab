"""
Basins are other .rtdc files on disk or online (DCOR, S3) which originate
from the same dataset but (potentially) contain more features. Basins
are useful if you would like to have a small copy of selected
features in a separate file while still being able to access
all features from the original file. E.g. you could load a small
.rtdc file from your local disk and access the larger "image"
feature data from an S3 basin. Basins are active by default, which
means that if you load a dataset that defines basins and these
basins are available, they will be integrated seamlessly into your
analysis pipeline. You can find out which features originate from
other basins via the ``features_basin`` property of an
:class:`.RTDCBase` instance.
"""
from __future__ import annotations

import io
import json
import pathlib
from typing import Any, BinaryIO, Dict, Literal
import warnings

import h5py


class ExternalDataForbiddenError(BaseException):
    """Raised when a dataset contains external data

    External data are a security risk, because they could be
    used to access data that are not supposed to be accessed.
    This is especially critical when the data are accessed within
    a web server process (e.g. in DCOR).
    """
    pass


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


def assert_no_external(h5):
    """Raise ExternalDataForbiddenError if `h5` refers to external data"""
    has_ext, path_ext = check_external(h5)
    if has_ext:
        raise ExternalDataForbiddenError(
            f"Dataset {h5.file.filename} contains external data, but these "
            f"are not permitted for security reasons ({path_ext})!")


def combine_h5files(paths: list,
                    external: Literal["follow", "raise"] = "follow"):
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
    linked_features: list of str
        List of features taken from the basins
    """
    fd = io.BytesIO()
    linked_features = []
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
                    # Also, write logs/tables/... (anything that is
                    # not events) only once.
                    for group in h5:
                        if group != "events":
                            hv[group] = h5py.ExternalLink(str(pp), group)
                # Append features
                hve = hv.require_group("events")
                for feat in h5["events"]:
                    if feat not in hve:
                        linked_features.append(feat)
                        hve[feat] = h5py.ExternalLink(str(pp),
                                                      f"/events/{feat}")
    return fd, linked_features


def initialize_basin_flooded_h5file(
        h5path: str | pathlib.Path | BinaryIO,
        h5kwargs: Dict[str, Any] | None = None,
        external: Literal["follow", "raise"] = "follow",
):
    """Combine HDF5 files into one in-memory H5File if basins were defined

    .. versionadded:: 0.51.0

    If an .rtdc file has "basins" defined, then these are sought out and
    referenced via external links in an in-memory :class:`h5py.H5File`.
    The advantage of this approach is that it transparently integrates
    with dclab: No additional file format readers need to be implemented
    and data from S3 object stores can be attached. The returned
    :class:`h5py.File` object contains all features from the original
    file and from its upstream basins.

    Parmeters
    ---------
    h5path: str or pathlib.Path or file-like object
        Path to an '.rtdc' measurement file or a file-like object
    h5kwargs: dict
        Additional keyword arguments given to :class:`h5py.File`
    external: str
        Defines how external (links, binary, virtual) data in the original
        and the basin datasets should be handled. The default is to "follow"
        external datasets or links to external data. In a zero-trust
        context, you can set this to "raise" which will cause an
        :class:`.ExternalDataForbiddenError` exception when external data
        are encountered.

    Returns
    -------
    h5file: BinaryIO
        An HDF5 file opened in binary mode (ready to be passed to
        :class:`h5py.File`)
    features_basin: list of str
        Feature names that were added from basins. In the :class:`.RTDC_HDF5`
        constructor, this is set as the `features_basin` instance property.

    TODO
    ----
    - Support for S3 basins
    - Reference/Identifier check for `file` basins
    - Make used basins available to user
    - Definition of basins specifications (order/priority, text, type, ...)
    - Also return features that are accessed online-only in a seprate list
      so the RTDCBase class can exclude them from `features_loaded`?
    """
    if h5kwargs is None:
        h5kwargs = {}

    # Increase the read cache (which defaults to 1MiB), since
    # normally we have around 2.5MiB image chunks.
    h5kwargs.setdefault("rdcc_nbytes", 10 * 1024 ** 2)
    h5kwargs.setdefault("rdcc_w0", 0)

    h5init = h5py.File(h5path, mode="r", **h5kwargs)
    if external == "raise":
        assert_no_external(h5init)
    # Check whether there are supported basins
    basin_list = []
    for bk in sorted(h5init.get("basins", [])):  # priority via `sorted`
        bdat = list(h5init["basins"][bk])
        if isinstance(bdat[0], bytes):
            bdat = [bi.decode("utf") for bi in bdat]
        bdict = json.loads(" ".join(bdat))
        # Check whether this basin is supported and exists
        if bdict["type"] == "file":
            for pp in bdict["paths"]:
                pp = pathlib.Path(pp)
                # Check absolute and relative paths
                if pp.exists():
                    basin_list.append(pp)
                    break
                # Also check for relative paths
                rp = h5path.parent / pp
                if rp.exists():
                    basin_list.append(rp)
                    break
        else:
            warnings.warn(
                f"Encountered unsupported basin type '{bdict['type']}'!")
    features_basin = []
    if basin_list:
        # We cannot link to HDF5 files that are already open du to the way
        # HDF5 handles file locking internally. Thus, we close the file and
        # insert the original path first in the list.
        h5init.close()
        basin_list.insert(0, h5path)
        fd, linked_features = combine_h5files(paths=basin_list,
                                              external=external)
        features_basin += linked_features
        h5file = h5py.File(fd)
    else:
        # This is the simplest case (there is only one file and no basins).
        h5file = h5init
    return h5file, features_basin
