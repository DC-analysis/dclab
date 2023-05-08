"""Test dictionary format"""
import numpy as np

from dclab import new_dataset

import pytest

from helper_methods import example_data_dict


def test_contour_not_available():
    ddict = example_data_dict(size=67, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    assert "contour" not in ds


def test_image_not_available():
    ddict = example_data_dict(size=67, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    assert "image" not in ds


def test_min_max_update():
    ddict = example_data_dict(size=67, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    cfg = ds.config.copy()

    cfg["filtering"]["deform min"] = .001
    cfg["filtering"]["deform max"] = .01
    ds.config.update(cfg)
    ds.apply_filter()

    assert ds.filter.all.sum() == 22


def test_readonly():
    ddict = example_data_dict(size=67, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    with pytest.raises(ValueError,
                       match="assignment destination is read-only"):
        ds["area_um"][:10] = 1


def test_shape_dict_contour():
    ddict = example_data_dict(size=67, keys=["area_um", "deform", "contour"])
    ds = new_dataset(ddict)
    assert ds["contour"].shape == (67, np.nan, 2)
    assert len(ds["contour"]) == 67


def test_shape_dict_trace():
    ddict = example_data_dict(size=67, keys=["area_um", "deform", "trace"])
    ds = new_dataset(ddict)
    assert ds["trace"].shape == (6, 67, 100)
    assert len(ds["trace"]) == 6
    assert len(ds["trace"]["fl1_raw"]) == 67
    assert len(ds["trace"]["fl1_raw"][0]) == 100


def test_trace_not_available():
    ddict = example_data_dict(size=67, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    assert "trace" not in ds


def test_wrong_things():
    ddict = example_data_dict(size=67, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    # Check unknown variable (warning will be displayed
    try:
        ds.apply_filter(force=["on_purpose_unknown"])
    except ValueError:
        pass


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
