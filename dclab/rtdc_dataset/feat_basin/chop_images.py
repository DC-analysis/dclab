from __future__ import annotations

from multiprocessing.sharedctypes import Synchronized
from typing import TYPE_CHECKING, Literal

import h5py
import hdf5plugin
import numpy as np
from scipy import ndimage

from ..import writer


if TYPE_CHECKING:
    from ..core import RTDCBase


def check_for_keep(size, shape, dtype):
    """Return whether a shape/size/dtype is large enough for storage"""
    size_req = writer.RTDCWriter.get_best_nd_chunks(
        item_shape=shape,
        item_dtype=dtype,
    )[0]
    return size >= size_req


def determine_clustering(geometry, dtype) -> tuple[np.ndarray, int]:
    """Return integer array of clusters for storage with same shape"""
    clusters = np.zeros(len(geometry), dtype=np.uint64)
    indices = np.arange(len(clusters))
    cluster_id = 1

    # create a copy of the shapes (we will modify this array)
    shapes = np.copy(geometry[:, :2])

    join_shapes = False
    join_x = False

    while not np.all(clusters):
        # indices that are not assigned to a cluster yet
        indices_todo = indices[clusters == 0]
        # shapes that need assignment
        shapes_todo = shapes[indices_todo]

        # identify unique shapes
        if join_x:
            shapes_to_unique = shapes_todo[:, ::-1]
        else:
            shapes_to_unique = shapes_todo

        uniques, inverse, counts = np.unique(shapes_to_unique,
                                             axis=0,
                                             return_counts=True,
                                             return_inverse=True,
                                             )

        if join_x:
            uniques = uniques[:, ::-1]

        if np.sum(counts) < 100 or len(uniques) == 1:
            clusters[indices_todo] = cluster_id
            cluster_id += 1
            break

        if join_shapes:
            # join adjacent shapes
            join_x = not join_x

            # extend for odd number of shapes
            if len(counts) % 2:
                counts = np.pad(counts, (0, 1))
                uniques = np.pad(uniques, ((0, 1), (0, 0)))
                uniques[-1] = uniques[-2]

            # update shapes
            for ii in range(0, len(uniques), 2):
                shape_match = indices_todo[inverse == ii]
                shapes[shape_match] = uniques[ii+1]

            # combine adjacent shapes
            uniques = uniques[1::2]
            inverse = inverse // 2
            counts = np.sum(counts.reshape(-1, 2), axis=1)

        for idx, (shape, size) in enumerate(zip(uniques, counts)):
            if check_for_keep(size=size, shape=shape, dtype=dtype):
                # Assign a common cluster to these shapes
                clusters[indices_todo[inverse == idx]] = cluster_id
                cluster_id += 1

        join_shapes = True

    return clusters, cluster_id - 1


def disk(radius):
    """Generates a flat, disk-shaped footprint.

    Taken from scikit-image (originally BSD-3-Clause license)

    A pixel is within the neighborhood if the Euclidean distance between
    it and the origin is no greater than radius.

    Parameters
    ----------
    radius : int
        The radius of the disk-shaped footprint.

    Returns
    -------
    footprint : ndarray
        The footprint where elements of the neighborhood are 1 and 0 otherwise.
    """
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    return (X**2 + Y**2) <= radius**2


def obtain_event_geometry(ds: RTDCBase,
                          pad_um: float = 1.5,
                          ) -> np.ndarray:
    """Return a shape array for all events in `ds`

    Parameters
    ----------
    ds:
        dataset to compute the shapes for
    pad_um:
        padding to add (top, left, right, bottom) to shape in [µm]

    Returns
    -------
    geometry
        2D numpy array of shape `(len(ds), 4)`.
        1st column is size along "y" axis.
        2nd column is size along "x" axis.
        3rd column is offset in "y".
        4th column is offset in "x".
    """
    geometry = np.zeros((len(ds), 4), dtype=np.uint16)
    pixel_size = ds.config["imaging"]["pixel size"]
    shape_y = ds.config["imaging"]["roi size y"]
    shape_x = ds.config["imaging"]["roi size x"]

    # use existing size feature for bounding box
    geometry[:, 0] = np.astype(
        np.ceil((ds["size_y"] + 2*pad_um) / pixel_size),
        np.int64)
    geometry[:, 1] = np.astype(
        np.ceil((ds["size_x"] + 2*pad_um) / pixel_size),
        np.int64)

    for mslice in ds["mask"].iter_chunks(10*1024**2):
        mask = ds["mask"][mslice]

        # determine position from mask data
        mask_y = np.count_nonzero(mask, axis=2) > 0
        start_y = np.argmax(mask_y, axis=1)  # first non-zero value along y
        geometry[mslice, 2] = np.clip(start_y - np.round(pad_um / pixel_size),
                                      a_min=0, a_max=shape_y)

        mask_x = np.count_nonzero(mask, axis=1) > 0
        start_x = np.argmax(mask_x, axis=1)  # first non-zero value along x
        geometry[mslice, 3] = np.clip(start_x - np.round(pad_um / pixel_size),
                                      a_min=0, a_max=shape_x)

    # correct elements with incorrect height
    too_high = geometry[:, 2] + geometry[:, 0] > shape_y
    geometry[too_high, 2] = shape_y - geometry[too_high, 0]

    # correct elements with incorrect width
    too_wide = geometry[:, 3] + geometry[:, 1] > shape_x
    geometry[too_wide, 3] = shape_x - geometry[too_wide, 1]
    return geometry


def write_chopped_images(
        ds: RTDCBase,
        feat: str,
        h5_dst: h5py.File,
        pad_um: float = 1.5,
        bytes_chopped: Synchronized[int] | None = None,
        crop_method: Literal["box", "dilation"] = "dilation",
        ) -> h5py.Group:
    """Write image feature from `ds` to chopped image data in `h5_dst`

    This method writes the feature specified as chopped image data
    to the output file.

    Parameters
    ----------
    ds:
        RTDCBase dataset containing the images
    feat:
        Which feature to chop. Must be in :const:`FEATURES_IMAGE_ROI`.
    h5_dst:
        Target file to write the feature data to
    pad_um:
        padding to add (top, left, right, bottom) to shape in [µm]
    bytes_chopped:
        multiprocessing value for tracking export process
    crop_method:
        method for cropping out the images; "box" means a rectangular
        bounding box, "dilation" means binary dilation which reduces
        the resulting file size by ~9%

    Returns
    -------
    h5_dst_group:
        HDF5 group containing the chopped image data

    Notes
    -----
    Traditionally, only the "image" feature is chopped. This algorithm
    should also work for "qpi_pha" and "qpi_amp". Don't use it for
    "image_bg" feature, because it is used for recovering "image" data.
    Chopping up the "mask" feature is supported, but discouraged, because
    it is compute intensive and yields only <10MB smaller file sizes.
    """
    # create output group
    h5_dst_group = h5_dst.require_group(f"/basin_events/{feat}")

    # Event geometry (shape and offset)
    geometry = obtain_event_geometry(ds=ds, pad_um=pad_um)
    feat_data = ds[feat]
    imshape = ds[feat].shape[1:]
    feat_dtype = feat_data.dtype
    is_boolean = feat_dtype == bool
    if is_boolean:
        # store as uin8, but treat as bool when loading
        h5_dst_group.attrs["is_boolean"] = True
        h5_feat_dtype = np.uint8
        crop_method = "box"  # because it is a boolean mask
    else:
        h5_feat_dtype = feat_dtype

    if feat == "image":
        bg_data = ds["image_bg"]
        h5_dst_group.attrs["inverse_background_feature"] = "image_bg"
    else:
        bg_data = None

    dilate = crop_method == "dilation"
    mask_data = ds["mask"]
    pixel_size = ds.config["imaging"]["pixel size"]
    # The radius of the disk for dilation is equal to the padding size.
    dilate_structure = disk(int(np.ceil(pad_um / pixel_size)))

    # Clusters (event indices that are written to the same dataset)
    clusters, num_clusters = determine_clustering(
        geometry=geometry, dtype=feat_dtype)

    # index array
    index = np.zeros((len(ds), 5), dtype=int)
    # offy
    index[:, 2] = geometry[:, 2]
    # offx
    index[:, 3] = geometry[:, 3]
    # other
    index[:, 4] = np.arange(len(index))
    # For the "mask" feature, all chopped images must be unique.
    # For all other features, add references to the sibling events
    # within one frame.
    if feat != "mask":
        frame = ds["frame"]
        _, fr_index, fr_counts = np.unique(frame,
                                           return_index=True,
                                           return_counts=True,
                                           )
        # We are only interested in frames with multiple events
        fr_relevant = fr_counts > 1
        fr_index = fr_index[fr_relevant]
        fr_counts = fr_counts[fr_relevant]
        # Set the indices
        for ii in range(len(fr_index)):
            # We assume that events with identical frames are monotonous
            same = np.arange(fr_index[ii], fr_index[ii] + fr_counts[ii])
            for idx in range(len(same)):
                jj = same[idx]
                index[jj, 4] = same[idx - 1]

    compression_kwargs = hdf5plugin.Zstd(clevel=5)

    # Prepare each cluster for writing
    for cid in range(1, num_clusters + 1):
        cidx = clusters == cid
        cidx_where = np.where(cidx)[0]
        # determine maximum shape
        shy, shx = np.max(geometry[cidx, :2], axis=0)
        # determine chunk size
        chunk_size_opt = writer.RTDCWriter.get_best_nd_chunks(
            item_shape=(shy, shx), item_dtype=feat_dtype)[0]
        num_events = len(cidx_where)
        chunk_size_opt = min(chunk_size_opt, num_events)
        num_chunks = int(np.ceil(num_events / chunk_size_opt))

        idx = 0  # enumerates events within cluster

        dset = h5_dst_group.create_dataset(
            name=f"{cid - 1}",
            shape=(0, shy, shx),
            dtype=h5_feat_dtype,
            maxshape=(num_events, shy, shx),
            chunks=(chunk_size_opt, shy, shx),
            fletcher32=True,
            **compression_kwargs)
        dset.attrs.create('CLASS', np.bytes_('IMAGE'))
        dset.attrs.create('IMAGE_VERSION', np.bytes_('1.2'))
        dset.attrs.create('IMAGE_SUBCLASS', np.bytes_('IMAGE_GRAYSCALE'))

        for _ in range(num_chunks):
            # assemble a chunk
            chunk_size = min(chunk_size_opt, num_events - idx)
            chunk_data = np.zeros((chunk_size, shy, shx), dtype=h5_feat_dtype)
            if dilate:
                # prepare mask data array for dilation
                chunk_mask = np.zeros((chunk_size, shy, shx), dtype=bool)
            else:
                chunk_mask = None

            for ii in range(chunk_size):
                ida = cidx_where[idx]
                shyi, shxi, offy, offx = geometry[ida][:4]
                if shyi + offy > imshape[0]:
                    offy = imshape[0] - shyi
                if shxi + offx > imshape[1]:
                    offx = imshape[1] - shxi
                feat_chop = feat_data[ida][offy:offy+shyi, offx:offx+shxi]
                if bg_data is not None:
                    feat_chop -= bg_data[ida][offy:offy+shyi, offx:offx+shxi]
                if dilate:
                    assert chunk_mask is not None
                    chunk_mask[ii, :shyi, :shxi] = \
                        mask_data[ida][offy:offy+shyi, offx:offx+shxi]
                chunk_data[ii, :shyi, :shxi] = feat_chop
                idx += 1

            if dilate:
                # run dilation on all masks
                mask_dilate = ndimage.binary_dilation(
                    input=chunk_mask,
                    structure=dilate_structure,
                    axes=(1, 2))
                # multiply chunk data with dilated mask
                chunk_data *= mask_dilate

            # write the chunk
            offset = dset.shape[0]
            dset.resize(offset + chunk_size, axis=0)
            if is_boolean:
                # Multiply mask feature with 255 so it is visible in HDFView
                chunk_data *= 255
            dset[offset:offset+chunk_size] = chunk_data
            # report the size of the original feature that we chopped down
            if bytes_chopped is not None:
                bytes_chopped.value += (
                    np.prod(imshape) * chunk_size * feat_dtype.itemsize)

        # update index
        # dataset
        index[cidx, 0] = cid - 1
        # sub-dataset index
        index[cidx, 1] = np.arange(num_events)

    # write index
    h5_dst_group.create_dataset(
        name="index",
        data=index,
        maxshape=index.shape,
        chunks=index.shape,
        fletcher32=True,
        **compression_kwargs)

    # write basin data
    with writer.RTDCWriter(h5_dst) as hw:
        hw.store_basin(
            basin_name=f"chopped-{feat}",
            basin_type="internal",
            basin_format="h5datasetchop",
            basin_locs=["basin_events"],
            basin_feats=[feat],
            internal_data=h5_dst_group,
        )

    return h5_dst_group
