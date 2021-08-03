import pathlib
import time

import h5py
import numpy as np
import pytest

import dclab
from dclab import new_dataset
from dclab.features.contour import get_contour, get_contour_lazily
from dclab.features.volume import get_volume

from scipy.ndimage import binary_fill_holes

from helper_methods import retrieve_data


def test_af_contour_basic():
    ds1 = dclab.new_dataset(retrieve_data("fmt-hdf5_mask-contour_2018.zip"))
    # export all data except for contour data
    features = ds1.features
    features.remove("contour")
    dspath = pathlib.Path(ds1.path)
    tempout = dspath.parent / (dspath.name + "without_contour.rtdc")
    ds1.export.hdf5(tempout, features=features)
    ds2 = dclab.new_dataset(tempout)

    for ii in range(len(ds1)):
        cin = ds1["contour"][ii]
        cout = ds2["contour"][ii]
        # simple presence test
        for ci in cin:
            assert ci in cout
        # order
        for jj in range(1, len(cin)):
            c2 = np.roll(cin, jj, axis=0)
            if np.all(c2 == cout):
                break
        else:
            assert False, "contours not matching, check orientation?"


def test_artefact():
    ds = new_dataset(retrieve_data("fmt-hdf5_fl-no-contour_2019.zip"))
    # This would raise a "dclab.features.contour.NoValidContourFoundError:
    # Event 1, No contour found!" in dclab version <= 0.22.1
    cont = ds["contour"][1]
    assert len(cont) == 37, "just to be sure there really is something"


def test_lazy_contour_basic():
    ds = new_dataset(retrieve_data("fmt-hdf5_mask-contour_2018.zip"))
    masks = ds["mask"][:]
    cont1 = get_contour_lazily(masks)
    cont2 = get_contour(masks)
    for ii in range(len(ds)):
        assert np.all(cont1[ii] == cont2[ii])


@pytest.mark.parametrize("idxs",
                         [slice(0, 5, 2),
                          [0, 2, 4],
                          np.array([0, 2, 4]),
                          np.array([True, False, True, False, True,
                                    False, False, False])
                          ])
def test_lazy_contour_slicing(idxs):
    h5path = retrieve_data("fmt-hdf5_mask-contour_2018.zip")

    contours_ref = []
    with h5py.File(h5path, "a") as h5:
        indices = np.arange(len(h5["events"]["deform"]))[idxs]
        assert np.all(indices == [0, 2, 4])
        for idn in indices:
            contours_ref.append(h5["events"]["contour"][str(idn)][:])
        del h5["events"]["contour"]
        masks_ref = np.array(h5["events"]["mask"][:][idxs], dtype=bool)

    with new_dataset(h5path) as ds:
        # sanity check
        assert "contour" not in ds.features_innate
        contours_test = ds["contour"][idxs]

    for csoll, cist, mask in zip(contours_ref, contours_test, masks_ref):
        # we cannot compare the contours directly, because they may
        # have different starting points. So we compare the masks.
        mask_ist = np.zeros_like(mask)
        mask_ist[cist[:, 1], cist[:, 0]] = True
        mask_ist = binary_fill_holes(mask_ist)

        assert np.all(mask_ist == mask)
        # remove them so we don't make any mistakes in defining
        # the next mask.
        del cist
        del mask_ist

        mask_soll = np.zeros_like(mask)
        mask_soll[csoll[:, 1], csoll[:, 0]] = True
        mask_soll = binary_fill_holes(mask_soll)
        assert np.all(mask_soll == mask)


def test_lazy_contour_single():
    ds = new_dataset(retrieve_data("fmt-hdf5_mask-contour_2018.zip"))
    masks = ds["mask"][:]
    c_all = get_contour_lazily(masks)
    c_0 = get_contour_lazily(masks[0])
    c_1 = get_contour_lazily(masks[1])
    assert np.all(c_0 == c_all[0])
    assert np.all(c_1 == c_all[1])
    assert len(c_0) != len(c_all[1])


def test_lazy_contour_timing():
    ds = new_dataset(retrieve_data("fmt-hdf5_mask-contour_2018.zip"))
    masks = ds["mask"][:]
    t0 = time.perf_counter()
    get_contour_lazily(masks)
    t1 = time.perf_counter()
    get_contour(masks)
    t2 = time.perf_counter()
    assert t2-t1 > 10*(t1-t0)


def test_lazy_contour_type():
    ds1 = new_dataset(retrieve_data("fmt-hdf5_mask-contour_2018.zip"))
    c1 = ds1["contour"]
    # force computation of contour data
    ds1._events._features.remove("contour")
    c2 = ds1["contour"]
    assert isinstance(c1, dclab.rtdc_dataset.fmt_hdf5.H5ContourEvent)
    assert isinstance(c2, dclab.features.contour.LazyContourList)


def test_simple_contour():
    pytest.importorskip("nptdms")
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image-bright_2017.zip"))
    # Note: contour "3" in ds is bad!
    cin = ds["contour"][2]
    mask = np.zeros_like(ds["image"][2], dtype="bool")
    mask[cin[:, 1], cin[:, 0]] = True
    cout = get_contour(mask)
    # length
    assert len(cin) == len(cout)
    # simple presence test
    for ci in cin:
        assert ci in cout
    # order
    for ii in range(1, len(cin)):
        c2 = np.roll(cin, ii, axis=0)
        if np.all(c2 == cout):
            break
    else:
        assert False, "contours not matching, check orientation?"


def test_volume():
    ds = new_dataset(retrieve_data("fmt-hdf5_mask-contour_2018.zip"))
    mask = [mi for mi in ds["mask"]]
    cont1 = [ci for ci in ds["contour"]]
    cont2 = get_contour(mask)

    kw = dict(pos_x=ds["pos_x"],
              pos_y=ds["pos_y"],
              pix=ds.config["imaging"]["pixel size"])

    v1 = get_volume(cont=cont1, **kw)
    v2 = get_volume(cont=cont2, **kw)

    assert np.allclose(v1, v2)
