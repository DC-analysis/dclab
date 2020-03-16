#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import time

import numpy as np
import pytest

import dclab
from dclab import new_dataset
from dclab.features.contour import get_contour, get_contour_lazily
from dclab.features.volume import get_volume

from helper_methods import retrieve_data, cleanup


def test_artefact():
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_mask_artefact.zip"))
    # This would raise a "dclab.features.contour.NoValidContourFoundError:
    # Event 1, No contour found!" in dclab version <= 0.22.1
    cont = ds["contour"][1]
    assert len(cont) == 37, "just to be sure there really is something"
    cleanup()


def test_lazy_contour_basic():
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_mask_contour.zip"))
    masks = ds["mask"][:]
    cont1 = get_contour_lazily(masks)
    cont2 = get_contour(masks)
    for ii in range(len(ds)):
        assert np.all(cont1[ii] == cont2[ii])
    cleanup()


@pytest.mark.skipif(sys.version_info < (3, 3),
                    reason="perf_counter requires python3.3 or higher")
def test_lazy_contour_timing():
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_mask_contour.zip"))
    masks = ds["mask"][:]
    t0 = time.perf_counter()
    get_contour_lazily(masks)
    t1 = time.perf_counter()
    get_contour(masks)
    t2 = time.perf_counter()
    assert t2-t1 > 10*(t1-t0)
    cleanup()


def test_lazy_contour_type():
    ds1 = new_dataset(retrieve_data("rtdc_data_hdf5_mask_contour.zip"))
    c1 = ds1["contour"]
    # force computation of contour data
    ds1._events._features.remove("contour")
    c2 = ds1["contour"]
    assert isinstance(c1, dclab.rtdc_dataset.fmt_hdf5.H5ContourEvent)
    assert isinstance(c2, dclab.features.contour.LazyContourList)
    cleanup()


def test_simple_contour():
    ds = new_dataset(retrieve_data("rtdc_data_traces_video_bright.zip"))
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
    cleanup()


def test_volume():
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_mask_contour.zip"))
    mask = [mi for mi in ds["mask"]]
    cont1 = [ci for ci in ds["contour"]]
    cont2 = get_contour(mask)

    kw = dict(pos_x=ds["pos_x"],
              pos_y=ds["pos_y"],
              pix=ds.config["imaging"]["pixel size"])

    v1 = get_volume(cont=cont1, **kw)
    v2 = get_volume(cont=cont2, **kw)

    assert np.allclose(v1, v2)
    cleanup()


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
