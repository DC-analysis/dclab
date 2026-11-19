"""Test dictionary format"""
import warnings

import numpy as np

from dclab import new_dataset, RTDCWriter
from dclab.rtdc_dataset.core import LocalBasinForbiddenWarning

import pytest

from helper_methods import example_data_dict


def test_contour_not_available():
    ddict = example_data_dict(size=67, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    assert "contour" not in ds


@pytest.mark.parametrize("features", [["trace", "deform"],
                                      ["deform", "trace"],
                                      ["trace"]])
def test_event_count_trace(features):
    """The event count must not be the number of trace channels"""
    data = {"deform": np.linspace(0.01, 0.1, 5),
            "trace": {"fl1_median": np.zeros((5, 10), dtype=np.int16),
                      "fl1_raw": np.zeros((5, 10), dtype=np.int16)}}
    ds = new_dataset({feat: data[feat] for feat in features})
    assert ds.config["experiment"]["event count"] == 5
    assert len(ds) == 5


def test_image_not_available():
    ddict = example_data_dict(size=67, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    assert "image" not in ds


def test_local_basins_allowed(tmp_path):
    """RTDC_Dict allows "file" basins (data are on the user's computer)"""
    path_basin = tmp_path / "basin.rtdc"
    with RTDCWriter(path_basin) as hw:
        hw.store_metadata({"experiment": {"sample": "basin",
                                          "run index": 1}})
        hw.store_feature("deform", np.linspace(0.01, 0.1, 10))
    ddict = {"area_um": np.linspace(10, 100, 10)}
    ds = new_dataset(ddict)
    bdict = {"name": "local file",
             "type": "file",
             "format": "hdf5",
             "paths": [str(path_basin)],
             "features": ["deform"],
             }
    # RTDC_Dict does not define basins itself
    ds.basins_get_dicts = lambda: [bdict]
    with warnings.catch_warnings():
        warnings.simplefilter("error", LocalBasinForbiddenWarning)
        assert "deform" in ds
    assert np.allclose(ds["deform"], np.linspace(0.01, 0.1, 10))


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
        if key.startswith("test_") and callable(loc[key]):
            loc[key]()
