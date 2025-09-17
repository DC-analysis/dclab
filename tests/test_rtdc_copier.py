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


def test_copy_basins():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_copy = path.with_name("test_copy.rtdc")

    # add log to source file
    with RTDCWriter(path,
                    mode="append",
                    compression_kwargs={"compression": None},
                    ) as hw:
        bn_hash = hw.store_basin(
            basin_type="file",
            basin_format="hdf5",
            basin_name="test_basin",
            basin_locs=["does-not-exist-but-does-not-matter.rtdc"],
            verify=False
        )
        assert not is_properly_compressed(hw.h5file[f"basins/{bn_hash}"])

    # copy
    with h5py.File(path) as h5, h5py.File(path_copy, "w") as hc:
        rtdc_copy(src_h5file=h5,
                  dst_h5file=hc)

    # Make sure this worked
    with h5py.File(path_copy) as hc:
        assert is_properly_compressed(hc["events/deform"])
        assert is_properly_compressed(hc[f"basins/{bn_hash}"])


def test_copy_basins_internal_none():
    """Store an internal basin and only export scalar features"""
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("input_file.rtdc")
    h5path_out = h5path.with_name("condensed.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_copy(src_h5file=src,
                  dst_h5file=hw.h5file,
                  features="scalar")
        assert "basins" not in hw.h5file, "no basins in input file"
        # store scalar and non-scalar internal basins in the input file
        hw.store_basin(basin_name="scalar and non-scalar basin data",
                       basin_type="internal",
                       basin_format="h5dataset",
                       basin_locs=["basin_events"],
                       basin_descr="an example test basin",
                       internal_data={"image_bg": np.zeros((2, 80, 320))},
                       basin_map=np.zeros(src["events/deform"].shape[0]),
                       basin_feats=["image_bg"],
                       )

    # sanity check
    with new_dataset(h5path_small) as ds:
        assert "image_bg" in ds.features_basin
        assert "image_bg" not in ds.features_innate

    # Now the actual tests starts.
    with h5py.File(h5path_small) as src, h5py.File(h5path_out, "a") as dst:
        rtdc_copy(src_h5file=src,
                  dst_h5file=dst,
                  features="scalar",
                  include_basins=True,
                  include_logs=True,
                  include_tables=True,
                  meta_prefix="")

    with h5py.File(h5path_out) as h5:
        # The output file should not contain the "image_bg" data.
        assert "basin_events" not in h5

    # When we load the basin data, "image_bg" should also not be there.
    with dclab.new_dataset(h5path_out) as ds:
        basin_dicts = ds.basins_get_dicts()
        assert len(basin_dicts) == 0


def test_copy_basins_internal_no_scalar():
    """Store an internal basin and only export scalar features"""
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("input_file.rtdc")
    h5path_out = h5path.with_name("condensed.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_copy(src_h5file=src,
                  dst_h5file=hw.h5file,
                  features="scalar")
        assert "basins" not in hw.h5file, "no basins in input file"
        # store scalar and non-scalar internal basins in the input file
        hw.store_basin(basin_name="scalar and non-scalar basin data",
                       basin_type="internal",
                       basin_format="h5dataset",
                       basin_locs=["basin_events"],
                       basin_descr="an example test basin",
                       internal_data={"userdef1": np.arange(2),
                                      "image_bg": np.zeros((2, 80, 320)),
                                      },
                       basin_map=np.zeros(src["events/deform"].shape[0]),
                       basin_feats=["image_bg", "userdef1"],
                       )

    # sanity check
    with new_dataset(h5path_small) as ds:
        assert "userdef1" in ds.features_basin
        assert "userdef1" not in ds.features_innate
        assert "image_bg" in ds.features_basin
        assert "image_bg" not in ds.features_innate

    # Now the actual tests starts.
    with h5py.File(h5path_small) as src, h5py.File(h5path_out, "a") as dst:
        rtdc_copy(src_h5file=src,
                  dst_h5file=dst,
                  features="scalar",
                  include_basins=True,
                  include_logs=True,
                  include_tables=True,
                  meta_prefix="")

    with h5py.File(h5path_out) as h5:
        # The output file should not contain the "image_bg" data.
        assert "userdef1" in h5["basin_events"]
        assert "image_bg" not in h5["basin_events"]

    # When we load the basin data, "image_bg" should also not be there.
    with dclab.new_dataset(h5path_out) as ds:
        basin_dicts = ds.basins_get_dicts()
        assert len(basin_dicts) == 1
        bn = basin_dicts[0]
        assert bn["type"] == "internal"
        assert bn["mapping"] != "same"
        assert bn["features"] == ["userdef1"], "image_bg feature must be gone"


def test_copy_basins_mapped():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_copy = path.with_name("test_copy.rtdc")

    # add log to source file
    with RTDCWriter(path,
                    mode="append",
                    compression_kwargs={"compression": None},
                    ) as hw:
        bn_hash = hw.store_basin(
            basin_type="file",
            basin_format="hdf5",
            basin_name="test_basin",
            basin_locs=["does-not-exist-but-does-not-matter.rtdc"],
            basin_map=np.arange(len(hw.h5file["events/deform"])),
            verify=False
        )
        assert not is_properly_compressed(hw.h5file[f"basins/{bn_hash}"])
        assert not is_properly_compressed(hw.h5file["events/basinmap0"])

    # copy
    with h5py.File(path) as h5, h5py.File(path_copy, "w") as hc:
        rtdc_copy(src_h5file=h5,
                  dst_h5file=hc)

    # Make sure this worked
    with h5py.File(path_copy) as hc:
        assert is_properly_compressed(hc["events/deform"])
        assert is_properly_compressed(hc[f"basins/{bn_hash}"])
        assert is_properly_compressed(hc["events/basinmap0"])


def test_copy_basins_multiple():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_copy = path.with_name("test_copy.rtdc")

    # add log to source file
    with RTDCWriter(path, mode="append") as hw:
        hw.store_basin(
            basin_type="file",
            basin_format="hdf5",
            basin_name="test_basin",
            basin_locs=["does-not-exist-but-does-not-matter.rtdc"],
            basin_map=np.arange(len(hw.h5file["events/deform"])),
            basin_descr="A first basin",
            verify=False
        )

        hw.store_basin(
            basin_type="file",
            basin_format="hdf5",
            basin_name="test_basin",
            basin_locs=["does-not-exist-but-does-not-matter.rtdc"],
            basin_map=np.arange(len(hw.h5file["events/deform"])),
            basin_descr="A second basin",
            verify=False
        )

        assert len(hw.h5file["basins"]) == 2

    # copy
    with h5py.File(path) as h5, h5py.File(path_copy, "w") as hc:
        rtdc_copy(src_h5file=h5,
                  dst_h5file=hc)

    # Make sure this worked
    with h5py.File(path_copy) as hc:
        assert len(hc["basins"]) == 2


def test_copy_basins_no_basin():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_copy = path.with_name("test_copy.rtdc")

    # add log to source file
    with RTDCWriter(path,
                    mode="append",
                    compression_kwargs={"compression": None},
                    ) as hw:
        bn_hash = hw.store_basin(
            basin_type="file",
            basin_format="hdf5",
            basin_name="test_basin",
            basin_locs=["does-not-exist-but-does-not-matter.rtdc"],
            basin_map=np.arange(len(hw.h5file["events/deform"])),
            verify=False
        )
        assert not is_properly_compressed(hw.h5file[f"basins/{bn_hash}"])
        assert not is_properly_compressed(hw.h5file["events/basinmap0"])

    # copy
    with h5py.File(path) as h5, h5py.File(path_copy, "w") as hc:
        rtdc_copy(src_h5file=h5,
                  dst_h5file=hc,
                  include_basins=False)

    # Make sure this worked
    with h5py.File(path_copy) as hc:
        assert is_properly_compressed(hc["events/deform"])
        assert "basins" not in hc
        assert "basinmap0" not in hc["events"]


def test_copy_logs():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_copy = path.with_name("test_copy.rtdc")

    # add log to source file
    with RTDCWriter(path,
                    mode="append",
                    compression_kwargs={"compression": None},
                    ) as hw:
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


def test_copy_no_events():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_in = path.with_name("input.rtdc")
    path_copy = path.with_name("test_copy.rtdc")

    with RTDCWriter(path_in, mode="append") as hw, h5py.File(path) as h5:
        hw.h5file.attrs.update(h5.attrs)
        hw.store_log("test_log", ["hans", "peter", "und", "zapadust"])

    # copy
    with h5py.File(path_in) as h5, h5py.File(path_copy, "w") as hc:
        # make sure no events are there
        if "events" in h5:
            assert len(h5["events"]) == 0
        rtdc_copy(src_h5file=h5,
                  dst_h5file=hc)

    # Make sure this worked
    with h5py.File(path_copy) as hc:
        if "events" in hc:
            assert len(hc["events"]) == 0
        assert "test_log" in hc["logs"]


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


def test_copy_specified_feature_list():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_copy = path.with_name("test_copy.rtdc")

    # copy
    with h5py.File(path) as h5, h5py.File(path_copy, "w") as hc:
        # make sure image data is there
        assert "image" in h5["events"]
        assert "area_um" in h5["events"]
        assert "deform" in h5["events"]
        rtdc_copy(src_h5file=h5,
                  dst_h5file=hc,
                  features=["image", "deform"])

    # Make sure this worked
    with h5py.File(path_copy) as hc:
        assert "image" in hc["events"]
        assert "area_um" not in hc["events"]
        assert "deform" in hc["events"]


def test_copy_tables():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_copy = path.with_name("test_copy.rtdc")

    # generate a table
    columns = ["bread", "beer", "chocolate"]
    ds_dt = np.dtype({'names': columns,
                      'formats': [np.float64] * len(columns)})
    tab_data = np.zeros((10, len(columns)))
    tab_data[:, 0] = np.arange(10)
    tab_data[:, 1] = 1000
    tab_data[:, 2] = np.linspace(np.pi, 2 * np.pi, 10)
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
            np.pi, 2 * np.pi, 10))


def test_copy_tables_array_only():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_copy = path.with_name("test_copy.rtdc")

    # generate a table that consists of an array, not a dict-like object
    tab_data = np.random.random((1000, 300))

    # add table to source file
    with h5py.File(path, "a") as h5:
        h5tab = h5.require_group("tables")
        h5tab.create_dataset(name="random_data",
                             data=tab_data)
        assert not is_properly_compressed(h5["tables/random_data"])

    # copy
    with h5py.File(path) as h5, h5py.File(path_copy, "w") as hc:
        rtdc_copy(src_h5file=h5,
                  dst_h5file=hc)

    # Make sure this worked
    with h5py.File(path_copy) as hc:
        assert is_properly_compressed(hc["tables/random_data"])
        assert np.all(tab_data == hc["tables/random_data"])


def test_copy_tables_hdf5_issue_3214():
    """Checks for a bug in HDF5

    https://github.com/HDFGroup/hdf5/issues/3214
    """
    path = retrieve_data("fmt-hdf5_segfault-compound_2023.zip")
    path_copy = path.with_name("test_copy.rtdc")

    # This caused a segmentation fault when using h5py.h5o.copy
    # with "tables/cytoshot_monitor".
    with h5py.File(path) as h5, h5py.File(path_copy, "w") as hc:
        rtdc_copy(src_h5file=h5,
                  dst_h5file=hc)

    # Also make sure metadata are copied
    with h5py.File(path_copy) as hc:
        assert hc["tables"]["cytoshot_monitor"].attrs["COLOR_shift"]\
               == "#0e8f69"


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


def test_do_not_copy_features():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_copy = path.with_name("test_copy.rtdc")

    # copy
    with h5py.File(path) as h5, h5py.File(path_copy, "w") as hc:
        # make sure image data is there
        assert "image" in h5["events"]
        assert "deform" in h5["events"]
        rtdc_copy(src_h5file=h5,
                  dst_h5file=hc,
                  features="none")

    # Make sure this worked
    with h5py.File(path_copy) as hc:
        assert "events" not in hc


def test_copier_with_wrong_feature():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_copy = path.with_name("test_copy.rtdc")

    # copy
    with h5py.File(path) as h5, h5py.File(path_copy, "w") as hc:
        # make sure image data is there
        assert "image" in h5["events"]
        assert "deform" in h5["events"]
        with pytest.raises(ValueError,
                           match="must be either a list of feature names"):
            rtdc_copy(src_h5file=h5,
                      dst_h5file=hc,
                      features="invalid")
        assert "events" not in hc
