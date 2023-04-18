import hdf5plugin
import h5py
import numpy as np
import pytest

import dclab
from dclab import new_dataset
from dclab.rtdc_dataset import is_properly_compressed, rtdc_copy
from dclab.rtdc_dataset import RTDCWriter

from helper_methods import retrieve_data


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_copy_already_compressed():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_temp = path.with_name("test_compressed.rtdc")
    path_copy = path.with_name("test_copy.rtdc")

    ds1 = new_dataset(path)

    with RTDCWriter(path_temp,
                    compression_kwargs=hdf5plugin.Zstd(clevel=5)) as hw:
        hw.store_metadata({"setup": ds1.config["setup"],
                           "experiment": ds1.config["experiment"]})
        hw.store_feature("deform", ds1["deform"])
        hw.store_feature("image", ds1["image"])

        # sanity check
        assert is_properly_compressed(hw.h5file["events/deform"])
        assert is_properly_compressed(hw.h5file["events/image"])

        with h5py.File(path_copy, "w") as hc:
            rtdc_copy(src_h5file=hw.h5file,
                      dst_h5file=hc)

    # Make sure this worked
    with h5py.File(path_copy) as hc:
        assert is_properly_compressed(hc["events/deform"])
        assert is_properly_compressed(hc["events/image"])


def test_copy_with_compression():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_copy = path.with_name("test_copy.rtdc")

    with h5py.File(path) as h5, h5py.File(path_copy, "w") as hc:
        rtdc_copy(src_h5file=h5,
                  dst_h5file=hc)

    # Make sure this worked
    with h5py.File(path_copy) as hc:
        assert is_properly_compressed(hc["events/deform"])
        assert is_properly_compressed(hc["events/image"])


def test_copy_logs():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_copy = path.with_name("test_copy.rtdc")

    # add log to source file
    with RTDCWriter(path, mode="append") as hw:
        hw.store_log("test_log", ["hans", "peter", "und", "zapadust"])
        assert not is_properly_compressed(hw.h5file["logs/test_log"])

    # copy
    with h5py.File(path) as h5, h5py.File(path_copy, "w") as hc:
        rtdc_copy(src_h5file=h5,
                  dst_h5file=hc)

    # Make sure this worked
    with h5py.File(path_copy) as hc:
        assert is_properly_compressed(hc["events/deform"])
        assert is_properly_compressed(hc["events/image"])
        assert is_properly_compressed(hc["logs/test_log"])


def test_copy_logs_variable_length_string():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_copy = path.with_name("test_copy.rtdc")

    # add variable-length log to source file
    with h5py.File(path, "a") as h5:
        data = ["hans", "peter", "und", "zapadust"]
        h5.require_group("logs")
        h5["logs"]["var_log"] = data
        assert h5["logs"]["var_log"].dtype.kind == "O"
        assert h5["logs/var_log"].dtype.str == "|O"

    # copy
    with h5py.File(path) as h5, h5py.File(path_copy, "w") as hc:
        rtdc_copy(src_h5file=h5,
                  dst_h5file=hc)

    # Make sure this worked
    with h5py.File(path_copy) as hc:
        assert is_properly_compressed(hc["events/deform"])
        assert is_properly_compressed(hc["events/image"])
        assert is_properly_compressed(hc["logs/var_log"])
        assert hc["logs/var_log"].dtype.kind == "S"
        assert hc["logs/var_log"].dtype.str == "|S100"


def test_copy_tables():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_copy = path.with_name("test_copy.rtdc")

    # generate a table
    columns = ["bread", "beer", "chocolate"]
    ds_dt = np.dtype({'names': columns,
                      'formats': [float] * len(columns)})
    tab_data = np.zeros((10, len(columns)))
    tab_data[:, 0] = np.arange(10)
    tab_data[:, 1] = 1000
    tab_data[:, 2] = np.linspace(np.pi, 2*np.pi, 10)
    rec_arr = np.rec.array(tab_data, dtype=ds_dt)
    # sanity check
    assert np.all(rec_arr["bread"][:].flatten() == np.arange(10))
    assert np.all(rec_arr["beer"][:].flatten() == 1000)
    assert np.all(rec_arr["chocolate"][:].flatten() == np.linspace(
        np.pi, 2 * np.pi, 10))

    # add table to source file
    with h5py.File(path, "a") as h5:
        h5tab = h5.require_group("tables")
        h5tab.create_dataset(name="most_important_things",
                             data=rec_arr)
        assert not is_properly_compressed(h5["tables/most_important_things"])

    # copy
    with h5py.File(path) as h5, h5py.File(path_copy, "w") as hc:
        rtdc_copy(src_h5file=h5,
                  dst_h5file=hc)

    # Make sure this worked
    with h5py.File(path_copy) as hc:
        assert is_properly_compressed(hc["tables/most_important_things"])
        tab_data = hc["tables/most_important_things"]
        assert np.all(tab_data["bread"][:].flatten() == np.arange(10))
        assert np.all(tab_data["beer"][:].flatten() == 1000)
        assert np.all(tab_data["chocolate"][:].flatten() == np.linspace(
            np.pi, 2*np.pi, 10))


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_copy_metadata_config():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_copy = path.with_name("test_copy.rtdc")

    # copy
    with h5py.File(path) as h5, h5py.File(path_copy, "w") as hc:
        rtdc_copy(src_h5file=h5,
                  dst_h5file=hc)

    with dclab.new_dataset(path_copy) as ds:
        assert ds.config["experiment"]["sample"] == "background image example"


def test_copy_metadata_of_datasets():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_copy = path.with_name("test_copy.rtdc")

    # add log to source file
    with RTDCWriter(path, mode="append") as hw:
        hw.store_log("test_log", ["hans", "peter", "und", "zapadust"])
        hw.h5file["logs/test_log"].attrs["saint"] = "germain"

    # copy
    with h5py.File(path) as h5, h5py.File(path_copy, "w") as hc:
        rtdc_copy(src_h5file=h5,
                  dst_h5file=hc)

    # Make sure this worked
    with h5py.File(path_copy) as hc:
        assert is_properly_compressed(hc["logs/test_log"])
        assert hc["logs/test_log"].attrs["saint"] == "germain"


def test_copy_scalar_features_only():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_copy = path.with_name("test_copy.rtdc")

    # copy
    with h5py.File(path) as h5, h5py.File(path_copy, "w") as hc:
        # make sure image data is there
        assert "image" in h5["events"]
        rtdc_copy(src_h5file=h5,
                  dst_h5file=hc,
                  features="scalar")

    # Make sure this worked
    with h5py.File(path_copy) as hc:
        assert "image" not in hc["events"]
        assert "deform" in hc["events"]
