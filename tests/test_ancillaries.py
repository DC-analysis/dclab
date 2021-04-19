import h5py
import numpy as np
from unittest.mock import Mock

import dclab

from helper_methods import example_data_dict, retrieve_data, \
    example_data_sets


def test_af_0basic():
    ds = dclab.new_dataset(retrieve_data(example_data_sets[1]))
    for cc in ['fl1_pos',
               'frame',
               'size_x',
               'size_y',
               'contour',
               'area_cvx',
               'circ',
               'image',
               'trace',
               'fl1_width',
               'nevents',
               'pos_x',
               'pos_y',
               'fl1_area',
               'fl1_max',
               ]:
        assert cc in ds

    # ancillaries
    for cc in ["deform",
               "area_um",
               "aspect",
               "frame",
               "index",
               "time",
               ]:
        assert cc in ds


def test_af_0error():
    keys = ["circ"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    try:
        ds["unknown_column"]
    except KeyError:
        pass
    else:
        raise ValueError("Should have raised KeyError!")


def test_af_aspect():
    # Aspect ratio of the data
    ds = dclab.new_dataset(retrieve_data("rtdc_data_traces_video_bright.zip"))
    aspect = ds["aspect"]
    assert np.sum(aspect > 1) == 904
    assert np.sum(aspect < 1) == 48


def test_af_area_ratio():
    ds = dclab.new_dataset(retrieve_data("rtdc_data_traces_video.zip"))
    comp_ratio = ds["area_ratio"]
    # The convex area is always >= the raw area
    assert np.all(comp_ratio >= 1)
    assert np.allclose(comp_ratio[0], 1.0196464)


def test_af_deform():
    keys = ["circ"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    assert np.allclose(ds["deform"], 1 - ds["circ"])


def test_af_populated_by_shared_method_hdf5():
    """Calling ancillary features will automatically populate other features
    that share the same method.
    """
    from dclab.rtdc_dataset.ancillaries.af_image_contour import (
        compute_bright as cb)
    cb.calls = 0
    path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    with h5py.File(path, "r+") as h5:
        _ = h5["events"]["bright_avg"][:]
        _ = h5["events"]["bright_sd"][:]
        del h5["events"]["bright_avg"]
        del h5["events"]["bright_sd"]
    ds = dclab.new_dataset(path)
    # sanity checks
    assert "bright_avg" not in ds.features_innate
    assert "bright_sd" not in ds.features_innate
    assert cb.calls == 0
    assert len(ds._ancillaries) == 0
    _ = ds["bright_avg"]
    # both "bright_avg" and "bright_sd" will be populated by one call
    assert cb.calls == 1
    assert len(ds._ancillaries) == 2
    feats = ["bright_sd", "bright_avg"]
    for af_key in ds._ancillaries:
        assert af_key in feats
    # clean up for further tests
    cb.calls = 0


def test_af_populated_by_shared_method_tdms():
    """Calling ancillary features will automatically populate other features
    that share the same method.
    """
    from dclab.rtdc_dataset.ancillaries.af_image_contour import (
        compute_bright as cb)
    cb.calls = 0
    ds = dclab.new_dataset(retrieve_data("rtdc_data_traces_video_bright.zip"))
    # This is something low-level and should not be done in a script.
    # Remove the brightness columns from RTDCBase to force computation with
    # the image and contour columns.
    _ = ds._events.pop("bright_avg")
    _ = ds._events.pop("bright_sd")
    assert "bright_avg" not in ds.features_innate
    assert "bright_sd" not in ds.features_innate
    assert cb.calls == 0
    assert len(ds._ancillaries) == 1  # time is populated
    _ = ds["bright_avg"]
    # both "bright_avg" and "bright_sd" will be populated by one call
    assert cb.calls == 1
    assert len(ds._ancillaries) == 3
    feats = ["time", "bright_sd", "bright_avg"]
    for af_key in ds._ancillaries:
        assert af_key in feats
    # clean up for further tests
    cb.calls = 0


def test_af_get_bright_called_only_once(monkeypatch):
    """Test that `get_bright` is only called once, which is desired when
    creating ancillary features that share the same pipeline"""
    def wrapper_on_get_bright(mm):
        dclab.features.bright.get_bright(
            mask=mm["mask"], image=mm["image"], ret_data="avg,sd")

    mock_run = Mock()
    monkeypatch.setattr('dclab.features.bright.get_bright', mock_run)

    ds = dclab.new_dataset(retrieve_data("rtdc_data_traces_video_bright.zip"))
    wrapper_on_get_bright(ds)

    mock_run.assert_called_once()
    mock_run.reset_mock()


def test_af_time():
    ds = dclab.new_dataset(retrieve_data("rtdc_data_minimal.zip"))
    tt = ds["time"]
    assert tt[0] == 0
    assert np.allclose(tt[1], 0.0385)
    assert np.all(np.diff(tt) > 0)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
