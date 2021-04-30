import pathlib
import numpy as np
import pytest

import dclab
from dclab.rtdc_dataset.plugins.plugin_feature import (
    PlugInFeature, find_plugin_feature_script, remove_all_plugin_features)

from helper_methods import retrieve_data


@pytest.fixture(autouse=True)
def cleanup_plugin_features():
    """Fixture used to setup and cleanup some fake ancillary features"""
    # code run before the test
    pass
    # then the test is run
    yield
    # code run after the test
    # remove our test plugin examples
    remove_all_plugin_features()


def get_plugin_file(plugin_name="plugin_test_example.py"):
    plugin_path = pathlib.Path(__file__).parent / "data" / plugin_name
    return plugin_path


def test_create_plugin():
    plugin_path = get_plugin_file()
    plugin_list = dclab.create_new_plugin_feature(plugin_path)
    assert isinstance(plugin_list[0], PlugInFeature)
    assert isinstance(plugin_list[1], PlugInFeature)

    ds = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
    assert "circ_per_area" in ds
    assert "circ_times_area" in ds
    circ_per_area = ds["circ_per_area"]
    circ_times_area = ds["circ_times_area"]
    assert np.allclose(circ_per_area, ds["circ"] / ds["area_um"])
    assert np.allclose(circ_times_area, ds["circ"] * ds["area_um"])


def test_remove_all_plugin_features():
    plugin_path = get_plugin_file()
    _ = dclab.create_new_plugin_feature(plugin_path)

    ds = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
    assert "circ_per_area" in ds
    assert "circ_times_area" in ds

    remove_all_plugin_features()

    assert "circ_per_area" not in ds
    assert "circ_times_area" not in ds


def test_plugin_metadata():
    plugin_path = get_plugin_file()
    plugin_list = dclab.create_new_plugin_feature(plugin_path)
    pf1, pf2 = plugin_list

    plugin_file_info = find_plugin_feature_script(plugin_path)

    assert pf1.plugin_metadata == plugin_file_info
    assert pf2.plugin_metadata == plugin_file_info


def compute_single_plugin_feature(rtdc_ds):
    circ_per_area = rtdc_ds["circ"] / rtdc_ds["area_um"]
    return circ_per_area


def compute_multiple_plugin_features(rtdc_ds):
    circ_per_area = rtdc_ds["circ"] / rtdc_ds["area_um"]
    circ_times_area = rtdc_ds["circ"] * rtdc_ds["area_um"]
    return {"circ_per_area": circ_per_area, "circ_times_area": circ_times_area}


def test_single_plugin_feature():
    ds = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))

    other_info = {}
    feature_info = {
        "feature_name": "circ_per_area",
        "method": compute_single_plugin_feature,
        "req_config": [],
        "req_features": ["circ", "area_um"],
        "priority": 1,
    }
    _ = PlugInFeature(other_info, **feature_info)
    assert "circ_per_area" in ds

    circ_per_area = ds["circ_per_area"]
    assert np.allclose(circ_per_area, ds["circ"] / ds["area_um"])

    # check that PlugInFeature exists independent of loaded ds
    with pytest.raises(AssertionError):
        ds2 = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
        assert "circ_per_area" not in ds2


def test_multiple_plugin_features():
    ds = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))

    other_info = {}
    feature_names = ["circ_per_area", "circ_times_area"]
    plugin_list = []
    for feature_name in feature_names:
        feature_info = {
            "feature_name": feature_name,
            "method": compute_multiple_plugin_features,
            "req_config": [],
            "req_features": ["circ", "area_um"],
            "priority": 1,
        }
        plugin_list.append(PlugInFeature(other_info, **feature_info))

    assert "circ_per_area" in ds
    assert "circ_times_area" in ds

    circ_per_area = ds["circ_per_area"]
    circ_times_area = ds["circ_times_area"]
    assert np.allclose(circ_per_area, ds["circ"] / ds["area_um"])
    assert np.allclose(circ_times_area, ds["circ"] * ds["area_um"])


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
            remove_all_plugin_features()
