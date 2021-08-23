""" Test functions for loading contours
"""

import os
import tempfile
import pathlib
import warnings
import pytest

import numpy as np
import h5py

from dclab.rtdc_dataset import new_dataset
import dclab.rtdc_dataset.config as dccfg
from test_rtdc_fmt_dcor import DCOR_AVAILABLE

from helper_methods import retrieve_data, example_data_dict


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


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_config_basic():
    pytest.importorskip("nptdms")
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))
    assert ds.config["imaging"]["roi size y"] == 96.


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_config_invalid_key():
    pytest.importorskip("nptdms")
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))
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
    pytest.importorskip("nptdms")
    # Download and extract data
    tdms_path = retrieve_data("fmt-tdms_minimal_2016.zip")
    ds = new_dataset(tdms_path)
    cfg_file = tempfile.mktemp(prefix="test_dclab_rtdc_config_")
    ds.config.save(cfg_file)
    loaded = dccfg.Configuration(files=[cfg_file])
    assert equals(loaded, ds.config)
    try:
        os.remove(cfg_file)
    except OSError:
        pass


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_config_update():
    pytest.importorskip("nptdms")
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))
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


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_user_section_allowed_key_types():
    """Check that the user config section keys only accept strings"""
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
    # strings are allowed
    ds.config["user"]["a string"] = "a string"
    # all other types will raise a BadUserConfigurationKeyWarning
    with pytest.warns(dccfg.BadUserConfigurationKeyWarning):
        ds.config["user"][True] = True
    with pytest.warns(dccfg.BadUserConfigurationKeyWarning):
        ds.config["user"][23.5] = 23.5
    with pytest.warns(dccfg.BadUserConfigurationKeyWarning):
        ds.config["user"][12] = 12
    with pytest.warns(dccfg.BadUserConfigurationKeyWarning):
        ds.config["user"][(5, 12.3, False, "a word")] = "a word"
    with pytest.warns(dccfg.BadUserConfigurationKeyWarning):
        ds.config["user"][[5, 12.3, False, "a word"]] = "a word"
    with pytest.warns(dccfg.BadUserConfigurationKeyWarning):
        ds.config["user"][{"name": 12.3, False: "a word"}] = "a word"
    with pytest.warns(dccfg.BadUserConfigurationValueWarning):
        ds.config["user"]["name"] = None

    assert len(ds.config["user"]) == 1
    assert isinstance(ds.config["user"]["a string"], str)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_user_section_basic():
    """Add information to the user section of config"""
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
    metadata = {"channel area": 100.5,
                "inlet": True,
                "n_constrictions": 3,
                "channel information": "other information"}
    ds.config.update({"user": metadata})
    assert ds.config["user"] == metadata


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_user_section_clear1():
    """Clear information from the user section with `clear()` method"""
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
    metadata = {"channel area": 100.5,
                "inlet": True,
                "n_constrictions": 3,
                "channel information": "other information"}
    ds.config.update({"user": metadata})
    assert ds.config["user"] == metadata
    ds.config["user"].clear()
    assert ds.config["user"] == {}


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_user_section_clear2():
    """Clear information from the user section with empty dict"""
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
    metadata = {"channel area": 100.5,
                "inlet": True,
                "n_constrictions": 3,
                "channel information": "other information"}
    ds.config.update({"user": metadata})
    assert ds.config["user"] == metadata
    ds.config["user"] = {}
    assert ds.config["user"] == {}


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_user_section_different_value_types():
    """Check that the user config section values take different data types"""
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
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


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_user_section_exists():
    """Check that the user config section exists"""
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
    assert ds.config["user"] == {}
    # control: nonsense sections don't exist
    with pytest.raises(KeyError):
        ds.config["Oh I seem to have lost my key"]


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_user_section_set_and_overwrite():
    """Add information to the user section of config via dict.__setitem__"""
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
    ds.config["user"]["some metadata"] = 42
    assert ds.config["user"] == {"some metadata": 42}
    metadata = {"more metadata": True}
    ds.config.update({"user": metadata})
    assert ds.config["user"] == {"some metadata": 42,
                                 "more metadata": True}
    # overwrite the previous keys and values
    ds.config["user"] = {}
    assert ds.config["user"] == {}


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_user_section_set_save_reload_empty_dict():
    """The 'user' config section as an empty dict will not save"""
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with new_dataset(h5path) as ds:
        ds.config.update({"user": {}})
        expath = h5path.with_name("exported.rtdc")
        ds.export.hdf5(expath, features=ds.features_innate)
    # nothing "user"-like is written to the HDF5 attributes
    with h5py.File(expath, "r") as h5:
        for ak in h5.attrs:
            assert not ak.startswith("user")
    # works for dclab because "user" added when checked
    with new_dataset(expath) as ds2:
        assert ds2.config["user"] == {}


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.parametrize("user_config",
                         [
                             {"": "a"},
                             {"": "b"},
                             {" ": "c"},
                             {"   ": "peter"},
                             {"\t": "pan"},
                             {"\n": "hook"},
                             {"\r": "croc"},
                         ])
def test_user_section_set_save_reload_empty_key(user_config):
    """Empty or only-whitespace 'user' section keys not allowed"""
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with new_dataset(h5path) as ds:
        with pytest.warns(dccfg.BadUserConfigurationKeyWarning):
            ds.config.update({"user": user_config})


def test_user_section_set_save_reload_fmt_dict():
    """Check that 'user' section metadata works for RTDC_Dict"""
    # create temp directory for storing outputted file
    tpath = pathlib.Path(tempfile.mkdtemp())
    ddict = example_data_dict(size=67, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    metadata = {"some metadata": 42}
    ds.config.update({"user": metadata})
    assert ds.config["user"] == metadata
    # must add some metadata to the "experiment" key for loading with dclab
    ds.config["experiment"]["sample"] = "test"
    ds.config["experiment"]["run index"] = 1
    expath = tpath / "exported.rtdc"
    with expath as exp:
        ds.export.hdf5(exp, features=["deform", "area_um"])
    # make sure that worked
    with h5py.File(expath, "r") as h5:
        assert h5.attrs["user:some metadata"] == 42
    # check again with dclab
    with new_dataset(expath) as ds2:
        assert ds2.config["user"] == metadata


@pytest.mark.skipif(not DCOR_AVAILABLE, reason="DCOR not reachable!")
def test_user_section_set_save_reload_fmt_dcor():
    """Check that 'user' section metadata works for RTDC_DCOR"""
    # create temp directory for storing outputted file
    tpath = pathlib.Path(tempfile.mkdtemp())
    with new_dataset("fb719fb2-bd9f-817a-7d70-f4002af916f0") as ds:
        metadata = {"some metadata": 12}
        ds.config.update({"user": metadata})
        assert ds.config["user"] == metadata
        expath = tpath / "exported.rtdc"
        ds.export.hdf5(expath, features=["deform", "area_um"])
    # make sure that worked
    with h5py.File(expath, "r") as h5:
        assert h5.attrs["user:some metadata"] == 12
    # check it again with dclab
    with new_dataset(expath) as ds2:
        assert ds2.config["user"] == metadata


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_user_section_set_save_reload_fmt_hdf5_basic():
    """Check that 'user' section metadata works for RTDC_HDF5"""
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
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


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_user_section_set_save_reload_fmt_hdf5_containers():
    """Check that 'user' section metadata works for container data types"""
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    channel_area = [0, 100]
    inlet = (1, 20, 40)
    outlet = np.array(inlet)
    metadata = {"channel area": channel_area,
                "inlet": inlet,
                "outlet": outlet}
    with new_dataset(h5path) as ds:
        ds.config.update({"user": metadata})
        expath = h5path.with_name("exported.rtdc")
        ds.export.hdf5(expath, features=ds.features_innate)
    # make sure that worked
    with h5py.File(expath, "r") as h5:
        assert all(h5.attrs["user:channel area"] == channel_area)
        assert all(h5.attrs["user:inlet"] == inlet)
        assert all(h5.attrs["user:outlet"] == outlet)
    # now check again with dclab
    with new_dataset(expath) as ds2:
        for k1 in metadata:
            assert all(metadata[k1] == ds2.config["user"][k1])


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_user_section_set_save_reload_fmt_hierarchy():
    """Check that 'user' section metadata works for RTDC_Hierarchy"""
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    metadata = {"channel area": 100.5,
                "inlet": True,
                "n_constrictions": 3,
                "channel information": "other information"}
    with new_dataset(h5path) as ds:
        ds.config.update({"user": metadata})
        ch = new_dataset(ds)
        expath = h5path.with_name("exported.rtdc")
        ch.export.hdf5(expath, features=ch.features_innate)
    # make sure that worked
    with h5py.File(expath, "r") as h5:
        assert h5.attrs["user:channel area"] == 100.5
        assert h5.attrs["user:inlet"]
        assert h5.attrs["user:n_constrictions"] == 3
        assert h5.attrs["user:channel information"] == "other information"
    # now check again with dclab
    with new_dataset(expath) as ds2:
        assert ds2.config["user"] == metadata


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.export.'
                            + 'LimitingExportSizeWarning')
@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.exc.'
                            + 'InitialFrameMissingWarning')
def test_user_section_set_save_reload_fmt_tdms():
    """Check that 'user' section metadata works for RTDC_TDMS"""
    pytest.importorskip("nptdms")
    h5path = retrieve_data("fmt-tdms_fl-image_2016.zip")
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


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_user_section_set_with_setitem():
    """Add information to the user section of config via dict.__setitem__"""
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
    ds.config["user"]["some metadata"] = 42
    assert ds.config["user"] == {"some metadata": 42}


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_user_section_set_with_update():
    """Add information to the user section of config with .update"""
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
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
