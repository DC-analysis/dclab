"""Helper methods for copying .rtdc data"""
import h5py
import h5py.h5o


class NotProperlyCompressedDatasetError(BaseException):
    """raised when data should be compressed but isn't"""


def h5copy(src_loc, src_name, dst_loc, ensure_compression=True):
    """Copy an HDF5 Dataset from one group to another

    Parameters
    ----------
    src_loc: h5py.H5Group
        The source location
    src_name: str
        Name of the dataset in `src_loc`
    dst_loc: h5py.H5Group
        The destination location
    ensure_compression: bool
        Whether to make sure that the data are compressed
        (disable if you are lazy)

    Raises
    ------
    NotProperlyCompressedDatasetError:
        If the source Dataset is not properly compressed, but
        `ensure_compression` is set
    ValueError:
        If the named source is not a h5py.Dataset
    """
    h5obj = src_loc[src_name]
    if isinstance(h5obj, h5py.Dataset):
        if ensure_compression and not is_properly_compressed(h5obj):
            raise NotProperlyCompressedDatasetError(
                f"The dataset {h5obj.name} in {h5obj.file} is not properly"
                f"compressed. Please disable `ensure_compression` or use "
                f"the `dclab.RTDCWriter` class instead."
            )
        # copy the Dataset to the destination
        h5py.h5o.copy(src_loc=src_loc.id,
                      src_name=src_name,
                      dst_loc=dst_loc,
                      dst_name=src_name,
                      )
    else:
        raise ValueError(f"The object {h5obj.name} in {h5obj.file} is not "
                         f"a dataset!")


def is_properly_compressed(h5obj):
    """Check whether an HDF5 object is properly compressed

    The compression check only returns True if the input file was
    compressed with the Zstandard compression using compression
    level 5 or higher.
    """
    # Since version 0.43.0, we use Zstandard compression
    # which does not show up in the `compression`
    # attribute of `obj`.
    create_plist = h5obj.id.get_create_plist()
    filter_args = create_plist.get_filter_by_id(32015)
    if filter_args is not None and filter_args[1][0] >= 5:
        properly_compressed = True
    else:
        properly_compressed = False
    return properly_compressed
