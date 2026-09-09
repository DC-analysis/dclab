import h5py
import numpy as np

from dclab import RTDCWriter, new_dataset
from dclab.rtdc_dataset.feat_basin import chop_images

from helper_methods import retrieve_data

import pytest


def create_fake_dataset(mult_factor=10):
    h5path = retrieve_data("fmt-hdf5_reference_2025.zip")
    h5path_large = h5path.with_name("large.rtdc")

    with (new_dataset(h5path) as ds0,
          RTDCWriter(h5path_large, compression_kwargs={}) as hw):
        # metadata
        hw.h5file.attrs.update(ds0.h5file.attrs)
        size = len(ds0)

        # make a large dataset
        for feat in ds0.features_innate:
            feat_data = np.concatenate([ds0[feat][:]] * mult_factor)

            if feat == "frame":
                feat_data += np.repeat(
                    size*np.arange(mult_factor, dtype=np.uint64), size)

            hw.store_feature(feat, feat_data)

    return h5path_large


@pytest.mark.parametrize("size,shape,keep", [
    [100, (80, 320), True],
    [10, (10, 10), False],
])
def test_check_for_keep(size, shape, keep):
    kwargs = {"size": size, "shape": shape, "dtype": np.uint8}
    assert chop_images.check_for_keep(**kwargs) == keep


def test_determine_clustering_simple():
    geometry = np.zeros((1000, 4), dtype=np.uint64)
    # 1st column is size along "y" axis.
    # 2nd column is size along "x" axis.
    # 3rd column is offset in "y".
    # 4th column is offset in "x".
    geometry[:, 0] = 43
    geometry[:, 1] = 66
    geometry[:100, 1] = 5

    clusters, num_clusters = chop_images.determine_clustering(
        geometry=geometry, dtype=np.uint8)

    assert num_clusters == 2
    assert np.all(clusters[100:] == 1)
    assert np.all(clusters[:100] == 2)


def test_determine_clustering_combine():
    geometry = np.zeros((50, 4), dtype=np.uint64)
    # 1st column is size along "y" axis.
    # 2nd column is size along "x" axis.
    # 3rd column is offset in "y".
    # 4th column is offset in "x".
    for ii in range(50):
        geometry[ii, 0] = 5 + ii
        geometry[ii, 1] = 4 + ii

    clusters, num_clusters = chop_images.determine_clustering(
        geometry=geometry, dtype=np.uint8)

    assert num_clusters == 1
    assert np.all(clusters == 1)


def test_chopped_image_base_small():
    h5path = create_fake_dataset(mult_factor=10)
    h5chop = h5path.with_name("chopped.rtdc")
    with new_dataset(h5path) as ds, RTDCWriter(h5chop) as hw:
        hw.h5file.attrs.update(ds.h5file.attrs)
        grp = chop_images.write_chopped_images(
            ds=ds,
            feat="image",
            h5_dst=hw.h5file,
        )
        assert "index" in grp
        assert "0" in grp

        index = grp["index"][:]
        assert np.all(index[:, 0] == 0)
        assert np.all(index[:, 1] == np.arange(len(index)))
        assert len(index) == 130


def test_chopped_image_base_large():
    h5path = create_fake_dataset(mult_factor=200)
    h5chop = h5path.with_name("chopped.rtdc")
    with new_dataset(h5path) as ds, RTDCWriter(h5chop) as hw:
        hw.h5file.attrs.update(ds.h5file.attrs)
        grp = chop_images.write_chopped_images(
            ds=ds,
            feat="image",
            h5_dst=hw.h5file,
        )

        hw.store_feature("frame", ds["frame"])
        hw.store_feature("time", ds["time"])
        hw.store_feature("image_bg", ds["image_bg"])
        hw.store_feature("mask", ds["mask"])

        assert "index" in grp
        assert "0" in grp
        assert "1" in grp

    with new_dataset(h5path) as ds0, new_dataset(h5chop) as dsc:
        # make sure all mask data match
        for ii in range(len(ds0)):
            # sanity check
            assert np.all(ds0["mask"][ii] == dsc["mask"][ii])
            # check image data
            assert np.all(ds0["image"][ii][ds0["mask"][ii]]
                          == dsc["image"][ii][dsc["mask"][ii]])

        # make sure whe have mask data in same-frame events
        assert ds0["frame"][1005] == dsc["frame"][1005] == 11092
        assert ds0["frame"][1006] == dsc["frame"][1006] == 11092
        assert ds0["frame"][1007] == dsc["frame"][1007] == 11092
        assert ds0["frame"][1008] == dsc["frame"][1008] == 11092

        for ii in range(1005, 1009):
            for jj in range(1005, 1009):
                assert np.all(ds0["image"][ii][ds0["mask"][ii]]
                              == dsc["image"][jj][dsc["mask"][ii]])


@pytest.mark.parametrize("mask_slice", [
    (slice(None, 10), slice(None, 10)),  # top left
    (slice(None, 10), slice(155, 165)),  # top center
    (slice(None, 10), slice(-10, None)),  # top right
    (slice(35, 45), slice(-10, None)),  # center right
    (slice(-10, None), slice(-10, None)),  # bottom right
    (slice(-10, None), slice(155, 165)),  # bottom center
    (slice(-10, None), slice(None, 10)),  # bottom left
    (slice(35, 45), slice(None, 10)),  # center left
])
def test_chopped_image_extremities(mask_slice):
    h5path = create_fake_dataset(mult_factor=50)

    # Add the fake event at position 500
    with h5py.File(h5path, "a") as h5:
        ps = h5.attrs["imaging:pixel size"]
        mask = np.zeros((80, 320), dtype=bool)
        mask[mask_slice] = True
        h5["events/mask"][500] = mask
        h5["events/size_x"][500] = 10 * ps
        h5["events/size_y"][500] = 10 * ps
        h5["events/pos_x"][500] = np.mean(np.arange(320)[mask_slice[1]]) * ps
        h5["events/pos_y"][500] = np.mean(np.arange(80)[mask_slice[0]]) * ps

    h5chop = h5path.with_name("chopped.rtdc")
    with new_dataset(h5path) as ds, RTDCWriter(h5chop) as hw:
        hw.h5file.attrs.update(ds.h5file.attrs)
        chop_images.write_chopped_images(
            ds=ds,
            feat="image",
            h5_dst=hw.h5file,
        )

        hw.store_feature("frame", ds["frame"])
        hw.store_feature("time", ds["time"])
        hw.store_feature("image_bg", ds["image_bg"])
        hw.store_feature("mask", ds["mask"])

    with new_dataset(h5path) as ds0, new_dataset(h5chop) as dsc:
        mask = np.zeros((80, 320), dtype=bool)
        mask[mask_slice] = True
        assert np.all(dsc["mask"][500] == mask)
        assert np.all(dsc["image"][500][mask] == ds0["image"][500][mask])


def test_chopped_image_and_mask():
    h5path = create_fake_dataset(mult_factor=200)
    h5chop = h5path.with_name("chopped.rtdc")
    with new_dataset(h5path) as ds, RTDCWriter(h5chop) as hw:
        hw.h5file.attrs.update(ds.h5file.attrs)
        chop_images.write_chopped_images(
            ds=ds,
            feat="image",
            h5_dst=hw.h5file,
        )
        grpm = chop_images.write_chopped_images(
            ds=ds,
            feat="mask",
            h5_dst=hw.h5file,
        )
        assert np.all(grpm["index"][:, 4] == np.arange(len(ds)))
        hw.store_feature("frame", ds["frame"])
        hw.store_feature("time", ds["time"])
        hw.store_feature("image_bg", ds["image_bg"])

    with new_dataset(h5path) as ds0, new_dataset(h5chop) as dsc:
        # make sure all mask data match
        for ii in range(len(ds0)):
            # Check whether the mask feature matches
            assert dsc["mask"][0].dtype == bool
            assert np.all(ds0["mask"][ii] == dsc["mask"][ii])
            # check image data
            assert np.all(ds0["image"][ii][ds0["mask"][ii]]
                          == dsc["image"][ii][dsc["mask"][ii]])
