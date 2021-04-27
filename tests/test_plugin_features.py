import numpy as np
import pytest

import dclab
from dclab.rtdc_dataset.plugins import PlugInFeature

from helper_methods import retrieve_data


@pytest.fixture
def cleanup_plugin_features():
    """Fixture used to setup and cleanup some fake ancillary features"""
    # code run before the test
    pass
    # then the test is run
    yield
    # code run after the test
    # remove our test plugin examples
    pass
    # remove_plugins()


def compute_single_plugin_feature(rtdc_ds):
    circ_per_area = rtdc_ds["circ"] / rtdc_ds["area_um"]
    return circ_per_area


def compute_multiple_plugin_features(rtdc_ds):
    circ_per_area = rtdc_ds["circ"] / rtdc_ds["area_um"]
    bright_per_area = rtdc_ds["bright_avg"] / rtdc_ds["area_um"]
    return {"circ_per_area": circ_per_area, "bright_per_area": bright_per_area}


def test_single_plugin_feature():
    ds = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))

    feature_info = {
        "feature_name": "circ_per_area",
        "method": compute_single_plugin_feature,
        "req_config": [],
        "req_features": ["circ", "area_um"],
        "priority": 1,
    }

    pf = PlugInFeature(**feature_info)
    assert pf.plugin_registered
    assert pf.plugin_info == feature_info
    circ_per_area = ds["circ_per_area"]
    assert np.allclose(circ_per_area, ds["circ"] / ds["area_um"])

    assert "circ_per_area" in ds
    with pytest.raises(AssertionError):
        ds2 = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
        assert "circ_per_area" not in ds2

    # ensure that the PlugInFeatures are no longer available
    PlugInFeature.features.remove(pf)
    PlugInFeature.feature_names.remove(pf.feature_name)
    ds3 = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
    assert "circ_per_area" not in ds3


def test_multiple_plugin_features():
    ds = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))

    feature_names = ["circ_per_area", "bright_per_area"]
    plugin_list = []
    for feature_name in feature_names:
        feature_info = {
            "feature_name": feature_name,
            "method": compute_multiple_plugin_features,
            "req_config": [],
            "req_features": ["circ", "area_um", "bright_avg"],
            "priority": 1,
        }
        plugin_list.append(PlugInFeature(**feature_info))

    _ = ds["circ_per_area"]
    _ = ds["bright_per_area"]

    assert "circ_per_area" in ds
    assert "bright_per_area" in ds

    # ensure that the PlugInFeatures are no longer available
    PlugInFeature.features.remove(plugin_list[0])
    PlugInFeature.feature_names.remove(plugin_list[0].feature_name)
    PlugInFeature.features.remove(plugin_list[1])
    PlugInFeature.feature_names.remove(plugin_list[1].feature_name)
    ds2 = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
    assert "circ_per_area" not in ds2
    assert "bright_per_area" not in ds2


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
            # use cleanup/deregister plugin function here also
