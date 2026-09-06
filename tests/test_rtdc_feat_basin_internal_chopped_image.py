import json

import h5py
import numpy as np

from dclab import rtdc_dataset, RTDCWriter, new_dataset
from dclab.rtdc_dataset.feat_basin import internal_chopped_image_basin

from helper_methods import retrieve_data


def make_chopped_image_dataset():
    """Return an HDF5 dataset that has a internal chopped image basin"""
    h5path = retrieve_data("fmt-hdf5_reference_2025.zip")
    h5path_chopped = h5path.with_name("chopped.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, h5py.File(h5path_chopped, "w") as dst:
        # metadata
        dst.attrs.update(src.attrs)

        # copy features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=dst,
                               features="scalar")
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=dst,
                               include_logs=False,
                               include_tables=False,
                               features=["mask", "image_bg"])

        # store the basin information in the new dataset
        bdat = {
            "type": "internal",
            "format": "h5datasetchop",
            "paths": ["basin_events"],
            "features": ["image"],
            "description": "chopped basin feature",
            "name": "chip and chap",
            "mapping": "same",
            "perishable": False,
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = dst.require_group("basins")
        with RTDCWriter(dst, mode="append") as hw:
            hw.write_text(basins, "chopper", blines)

        # prepare index dataset
        size = src.attrs["experiment:event count"]
        index = np.zeros((size, 5), dtype=int)

        # generate image basin data based on the original file
        bne = dst.require_group("basin_events")
        bim = bne.require_group("image")
        bim.attrs["inverse_background_feature"] = "image_bg"
        # create two datasets
        # manually prepared the indices that we want to put in each dataset
        idx0 = [3, 5, 7, 9, 10, 11, 12]
        idx1 = [0, 1, 2, 4, 6, 8]
        posx = src["events"]["pos_x"][:] / src.attrs["imaging:pixel size"]
        # shape is (80, 320)

        arr0 = np.zeros((len(idx0), 70, 80), dtype=np.uint8)
        arr1 = np.zeros((len(idx1), 80, 90), dtype=np.uint8)
        id0 = 0
        id1 = 0

        for ii in range(size):
            if ii in idx0:
                arr = arr0
                offy = 5
                slicey = slice(5, -5)
                did = 0
                iid = id0
                id0 += 1
            else:
                arr = arr1
                offy = 0
                slicey = slice(None, None)
                did = 1
                iid = id1
                id1 += 1

            shx = arr.shape[2]
            x0 = max(0, int(posx[ii] - shx // 2))
            x1 = min(320, int(posx[ii] + shx // 2))
            if x0 == 0:
                x1 = x0 + shx
            if x1 == 320:
                x0 = x1 - shx

            arr[iid] = (src["events/image"][ii][slicey, x0:x1]
                        - src["events/image_bg"][ii][slicey, x0:x1])

            index[ii] = [did, iid, offy, x0, ii]

        # add frame information
        frame = src["events/frame"][:]
        for fr in np.unique(frame):
            same = np.where(frame == fr)[0]
            if len(same) != 1:
                for idx in range(len(same)):
                    ii = same[idx]
                    index[ii, 4] = same[idx - 1]

        bim["0"] = arr0
        bim["1"] = arr1

        # write index dataset
        bim["index"] = index

        # debugging help
        for ii in "0", "1":
            bim[ii].attrs['CLASS'] = np.bytes_('IMAGE')
            bim[ii].attrs['IMAGE_VERSION'] = np.bytes_('1.2')
            bim[ii].attrs['IMAGE_SUBCLASS'] = np.bytes_('IMAGE_GRAYSCALE')

    return h5path, h5path_chopped


def test_chopped_basin_feature_access():
    """Make sure that the information inside the mask is identical"""
    path, path_chopped = make_chopped_image_dataset()

    # sanity check
    with h5py.File(path) as h5:
        assert "image" in h5["events"]

    with h5py.File(path_chopped) as h5:
        assert "image" not in h5["events"]
        assert "image" in h5["basin_events"]

    with new_dataset(path) as ds0, new_dataset(path_chopped) as dsc:
        for ii in range(len(ds0)):
            assert np.all(ds0["image"][ii][ds0["mask"][ii]]
                          == dsc["image"][ii][dsc["mask"][ii]])


def test_chopped_basin_feature_multiple_events_per_frame():
    """When there are multiple events in a frame, make sure all are shown"""
    path, path_chopped = make_chopped_image_dataset()

    with new_dataset(path) as ds0, new_dataset(path_chopped) as dsc:
        # In this dataset, we have multiple frames that contain multiple
        # events. Here we are testing frame 10091.
        assert ds0["frame"][4] == dsc["frame"][4] == 10091
        assert ds0["frame"][5] == dsc["frame"][5] == 10091
        assert ds0["frame"][6] == dsc["frame"][6] == 10091
        assert ds0["frame"][7] == dsc["frame"][7] == 10091

        for ii in range(4, 8):
            for jj in range(4, 8):
                assert np.all(ds0["image"][ii][ds0["mask"][ii]]
                              == dsc["image"][jj][dsc["mask"][ii]])


def test_chopped_basin_feature_proxy_array_indexing():
    path, path_chopped = make_chopped_image_dataset()

    # sanity check
    with new_dataset(path) as ds0:
        assert ds0["image"][:].shape == (13, 80, 320)

    with new_dataset(path_chopped) as dsc:
        image = dsc["image"]
        assert isinstance(
            image,
            internal_chopped_image_basin.InternalImageChoppedFeatureProxy)
        # This should never be done in production
        imdat = image[:]
        assert imdat.shape == (13, 80, 320)

        for ii in range(len(dsc)):
            assert np.all(dsc["image"][ii] == imdat[ii])


def test_chopped_basin_feature_proxy_array_indexing_array():
    _, path_chopped = make_chopped_image_dataset()

    indices = [1, 5, 10]

    with new_dataset(path_chopped) as dsc:
        image = dsc["image"]
        assert isinstance(
            image,
            internal_chopped_image_basin.InternalImageChoppedFeatureProxy)
        # This should never be done in production
        imdat = image[indices]
        assert imdat.shape == (3, 80, 320)

        for ii, idx in enumerate(indices):
            assert np.all(dsc["image"][idx] == imdat[ii])


def test_chopped_basin_feature_proxy_array_indexing_range():
    _, path_chopped = make_chopped_image_dataset()

    indices = range(1, 7)

    with new_dataset(path_chopped) as dsc:
        image = dsc["image"]
        assert isinstance(
            image,
            internal_chopped_image_basin.InternalImageChoppedFeatureProxy)
        # This should never be done in production
        imdat = image[indices]
        assert imdat.shape == (6, 80, 320)

        for ii, idx in enumerate(indices):
            assert np.all(dsc["image"][idx] == imdat[ii])


def test_chopped_basin_feature_proxy_array_method():
    path, path_chopped = make_chopped_image_dataset()

    # sanity check
    with new_dataset(path) as ds0:
        assert ds0["image"][:].shape == (13, 80, 320)

    with new_dataset(path_chopped) as dsc:
        assert dsc["image"].shape == (13, 80, 320)
        # This should never be done in production
        image = np.array(dsc["image"])
        assert isinstance(image, np.ndarray)
        assert image.shape == (13, 80, 320)

        for ii in range(len(dsc)):
            assert np.all(dsc["image"][ii] == image[ii])


def test_chopped_basin_feature_proxy_array_method_dtype():
    path, path_chopped = make_chopped_image_dataset()

    # sanity check
    with new_dataset(path) as ds0:
        assert ds0["image"][:].shape == (13, 80, 320)

    with new_dataset(path_chopped) as dsc:
        assert dsc["image"].shape == (13, 80, 320)
        # This should never be done in production
        image = np.array(dsc["image"], dtype=np.int64)
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.int64
        assert image.shape == (13, 80, 320)

        for ii in range(len(dsc)):
            assert np.all(dsc["image"][ii] == image[ii])


def test_chopped_basin_feature_proxy_array_mean():
    _, path_chopped = make_chopped_image_dataset()

    with new_dataset(path_chopped) as dsc:
        # This should never be done in production
        means = np.mean(dsc["image"], axis=(1, 2))
        assert means.shape == (13,)

        for ii in range(len(dsc)):
            assert np.allclose(means[ii], np.mean(dsc["image"][ii]),
                               rtol=0, atol=1e-12)


def test_chopped_basin_feature_proxy_array_ufunc():
    _, path_chopped = make_chopped_image_dataset()

    with new_dataset(path_chopped) as dsc:
        # This should never be done in production
        added = np.add(dsc["image"], 1)

        for ii in range(len(dsc)):
            assert np.allclose(added[ii], dsc["image"][ii] + 1,
                               rtol=0, atol=1e-12)
