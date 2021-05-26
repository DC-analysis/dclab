""" Test functions for loading contours
"""

import os
import tempfile
import warnings
import pytest

import numpy as np
import h5py

from dclab.rtdc_dataset import new_dataset
import dclab.rtdc_dataset.config as dccfg

from helper_methods import retrieve_data, example_data_sets


def equals(a, b):
    """Compare objects with allclose"""
    if isinstance(a, (dict, dccfg.Configuration, dccfg.ConfigurationDict)):
        for key in a:
            assert key in b, "key not in b"
            assert equals(a[key], b[key])
    elif isinstance(a, (float, int)):
        if np.isnan(a):
            assert np.isnan(b)
        else:
            assert np.allclose(a, b), "a={} vs b={}".format(a, b)
    else:
        assert a == b, "a={} vs b={}".format(a, b)
    return True


def test_config_basic():
    ds = new_dataset(retrieve_data(example_data_sets[1]))
    assert ds.config["imaging"]["roi size y"] == 96.


def test_config_invalid_key():
    ds = new_dataset(retrieve_data(example_data_sets[1]))
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        ds.config["setup"]["invalid_key"] = "picard"
        # Verify some things
        assert len(w) == 1
        assert issubclass(w[-1].category, dccfg.UnknownConfigurationKeyWarning)
        assert "invalid_key" in str(w[-1].message)


def test_config_save_load():
    # Download and extract data
    tdms_path = retrieve_data(example_data_sets[0])
    ds = new_dataset(tdms_path)
    cfg_file = tempfile.mktemp(prefix="test_dclab_rtdc_config_")
    ds.config.save(cfg_file)
    loaded = dccfg.Configuration(files=[cfg_file])
    assert equals(loaded, ds.config)
    try:
        os.remove(cfg_file)
    except OSError:
        pass


def test_config_update():
    ds = new_dataset(retrieve_data(example_data_sets[1]))
    assert ds.config["imaging"]["roi size y"] == 96.
    ds.config["calculation"].update({
        "crosstalk fl12": 0.1,
        "crosstalk fl13": 0.2,
        "crosstalk fl21": 0.3,
        "crosstalk fl23": 0.4,
        "crosstalk fl31": 0.5,
        "crosstalk fl32": 0.6,
    })

    assert ds.config["calculation"]["crosstalk fl12"] == 0.1
    assert ds.config["calculation"]["crosstalk fl13"] == 0.2
    assert ds.config["calculation"]["crosstalk fl21"] == 0.3
    assert ds.config["calculation"]["crosstalk fl23"] == 0.4
    assert ds.config["calculation"]["crosstalk fl31"] == 0.5
    assert ds.config["calculation"]["crosstalk fl32"] == 0.6


def test_user_section_basic():
    """Add information to the user section of config"""
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
    metadata = {"channel area": 100.5,
                "inlet": True,
                "n_constrictions": 3,
                "channel information": "other information"}
    ds.config.update({"user": metadata})
    assert ds.config["user"] == metadata


def test_user_section_different_key_types():
    """Check that the user config section keys take different data types"""
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
    assert ds.config["user"] == {}
    ds.config["user"][True] = True
    ds.config["user"][23.5] = 23.5
    ds.config["user"][12] = 12
    ds.config["user"]["a string"] = "a string"

    assert len(ds.config["user"]) == 4
    assert isinstance(ds.config["user"][True], bool)
    assert isinstance(ds.config["user"][23.5], float)
    assert isinstance(ds.config["user"][12], int)
    assert isinstance(ds.config["user"]["a string"], str)


def test_user_section_different_value_types():
    """Check that the user config section values take different data types"""
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
    assert ds.config["user"] == {}
    ds.config["user"]["channel A"] = True
    ds.config["user"]["channel B"] = 23.5
    ds.config["user"]["channel C"] = 12
    ds.config["user"]["channel D"] = "a string"
    ds.config["user"]["channel E"] = {"a key": 23}
    ds.config["user"]["channel F"] = [8, 55.4, True, "a word"]
    ds.config["user"]["channel G"] = (5, 12.3, False, "a word")

    assert len(ds.config["user"]) == 7
    assert isinstance(ds.config["user"]["channel A"], bool)
    assert isinstance(ds.config["user"]["channel B"], float)
    assert isinstance(ds.config["user"]["channel C"], int)
    assert isinstance(ds.config["user"]["channel D"], str)
    assert isinstance(ds.config["user"]["channel E"], dict)
    assert isinstance(ds.config["user"]["channel F"], list)
    assert isinstance(ds.config["user"]["channel G"], tuple)


def test_user_section_exists():
    """Check that the user config section exists"""
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
    assert ds.config["user"] == {}
    # show that nonsense sections don't exist
    with pytest.raises(KeyError):
        _ = ds.config["Oh I seem to have lost my key"]


def test_user_section_key_types_unhashable():
    """Show the user section keys that are not allowed (unhashable)"""
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
    with pytest.raises(TypeError):
        ds.config["user"][{"a key": 23}] = {"a key": 23}
        ds.config["user"][[8, 55.4, True, "a word"]] = "a word"
        ds.config["user"][(5, 12.3, False, "a word")] = "a word"


def test_user_section_set_save_reload():
    """Save the user section config information and reload it"""
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    metadata = {"channel area": 100.5,
                "inlet": True,
                "n_constrictions": 3,
                "channel information": "other information"}
    with new_dataset(h5path) as ds:
        ds.config.update({"user": metadata})
        expath = h5path.with_name("exported.rtdc")
        ds.export.hdf5(expath, features=ds.features_innate)
    # make sure that worked
    with h5py.File(expath, "r") as h5:
        assert h5.attrs["user:channel area"] == 100.5
        assert h5.attrs["user:inlet"]
        assert h5.attrs["user:n_constrictions"] == 3
        assert h5.attrs["user:channel information"] == "other information"
    # now check again with dclab
    with new_dataset(expath) as ds2:
        assert ds2.config["user"] == metadata


def test_user_section_set_with_setitem():
    """Add information to the user section of config via dict.__setitem__"""
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
    ds.config["user"]["some metadata"] = 42
    assert ds.config["user"] == {"some metadata": 42}


def test_user_section_set_with_update():
    """Add information to the user section of config with .update"""
    ds = new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
    metadata = {"some metadata": 42}
    ds.config.update({"user": metadata})
    assert ds.config["user"] == {"some metadata": 42}

    metadata2 = {"channel information": "information"}
    ds.config["user"].update(metadata2)
    assert ds.config["user"] == {"some metadata": 42,
                                 "channel information": "information"}


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
