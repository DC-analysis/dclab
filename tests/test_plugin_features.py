import pathlib
import numpy as np
import pytest

import dclab
from dclab.rtdc_dataset.plugins.plugin_feature import (
    PlugInFeature, import_plugin_feature_script,
    remove_plugin_feature, remove_all_plugin_features,
    PluginImportError,
)

from helper_methods import retrieve_data

data_dir = pathlib.Path(__file__).parent / "data"


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


def compute_single_plugin_feature(rtdc_ds):
    circ_per_area = rtdc_ds["circ"] / rtdc_ds["area_um"]
    return circ_per_area


def compute_multiple_plugin_features(rtdc_ds):
    circ_per_area = rtdc_ds["circ"] / rtdc_ds["area_um"]
    circ_times_area = rtdc_ds["circ"] * rtdc_ds["area_um"]
    return {"circ_per_area": circ_per_area, "circ_times_area": circ_times_area}


def test_import_plugin():
    plugin_path = data_dir / "plugin_test_example.py"
    plugin_list = dclab.load_plugin_feature(plugin_path)
    assert isinstance(plugin_list[0], PlugInFeature)
    assert isinstance(plugin_list[1], PlugInFeature)

    ds = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
    assert "circ_per_area" in ds
    assert "circ_times_area" in ds
    circ_per_area = ds["circ_per_area"]
    circ_times_area = ds["circ_times_area"]
    assert np.allclose(circ_per_area, ds["circ"] / ds["area_um"])
    assert np.allclose(circ_times_area, ds["circ"] * ds["area_um"])


def test_import_plugin_fail():
    bad_plugin_path = "not/a/real/path/plugin.py"
    with pytest.raises(PluginImportError):
        _ = dclab.load_plugin_feature(bad_plugin_path)


def test_remove_plugin_feature():
    plugin_path = data_dir / "plugin_test_example.py"
    plugin_list = dclab.load_plugin_feature(plugin_path)
    assert len(plugin_list) == 2

    ds = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
    assert "circ_per_area" in ds
    assert "circ_times_area" in ds
    assert dclab.dfn.feature_exists("circ_per_area")
    assert dclab.dfn.feature_exists("circ_times_area")

    remove_plugin_feature(plugin_list[0])
    remove_plugin_feature(plugin_list[1])

    assert "circ_per_area" not in ds
    assert "circ_times_area" not in ds
    assert not dclab.dfn.feature_exists("circ_per_area")
    assert not dclab.dfn.feature_exists("circ_times_area")

    with pytest.raises(TypeError):
        not_a_plugin_instance = [4, 6, 5]
        remove_plugin_feature(not_a_plugin_instance)


def test_remove_all_plugin_features():
    plugin_path = data_dir / "plugin_test_example.py"
    _ = dclab.load_plugin_feature(plugin_path)

    ds = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
    assert "circ_per_area" in ds
    assert "circ_times_area" in ds
    assert dclab.dfn.feature_exists("circ_per_area")
    assert dclab.dfn.feature_exists("circ_times_area")

    remove_all_plugin_features()

    assert "circ_per_area" not in ds
    assert "circ_times_area" not in ds
    assert not dclab.dfn.feature_exists("circ_per_area")
    assert not dclab.dfn.feature_exists("circ_times_area")


def test_plugin_attributes():
    plugin_path = data_dir / "plugin_test_example.py"
    plugin_list = dclab.load_plugin_feature(plugin_path)
    pf1, pf2 = plugin_list

    plugin_file_info = import_plugin_feature_script(plugin_path)

    assert pf1.plugin_path == plugin_path
    assert pf2.plugin_path == plugin_path
    assert pf1.plugin_info == plugin_file_info
    assert pf2.plugin_info == plugin_file_info
    assert pf1.feature_label == plugin_file_info["feature labels"][0]
    assert pf2.feature_label == plugin_file_info["feature labels"][1]
    assert pf1.is_scalar
    assert pf2.is_scalar


def test_single_plugin_feature():
    ds = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))

    plugin_path = ''
    other_info = {}
    feature_label = "Circularity per Area"
    is_scalar = True
    feature_info = {
        "feature_name": "circ_per_area",
        "method": compute_single_plugin_feature,
        "req_config": [],
        "req_features": ["circ", "area_um"],
        "req_func": lambda x: True,
        "priority": 1,
    }
    _ = PlugInFeature(feature_label, is_scalar,
                      plugin_path, other_info, **feature_info)
    assert "circ_per_area" in ds

    circ_per_area = ds["circ_per_area"]
    assert np.allclose(circ_per_area, ds["circ"] / ds["area_um"])

    # check that PlugInFeature exists independent of loaded ds
    with pytest.raises(AssertionError):
        ds2 = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
        assert "circ_per_area" not in ds2


def test_multiple_plugin_features():
    ds = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))

    plugin_path = ''
    other_info = {}
    feature_names = ["circ_per_area", "circ_times_area"]
    feature_labels = ["Circularity per Area", "Circularity times Area"]
    is_scalar = True
    plugin_list = []
    for feature_name, feature_label in zip(feature_names, feature_labels):
        feature_info = {
            "feature_name": feature_name,
            "method": compute_multiple_plugin_features,
            "req_config": [],
            "req_features": ["circ", "area_um"],
            "req_func": lambda x: True,
            "priority": 1,
        }
        plugin_list.append(PlugInFeature(
            feature_label, is_scalar, plugin_path, other_info, **feature_info))

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
