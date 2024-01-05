import json

import h5py
import numpy as np

import pytest

import dclab
from dclab import new_dataset, rtdc_dataset, RTDCWriter
from dclab.rtdc_dataset import feat_basin


from helper_methods import retrieve_data


def test_basin_sorting_basic():
    bnlist = [
        {"type": "remote", "format": "dcor", "ident": 0},
        {"type": "file", "format": "hdf5", "ident": 1},
        {"type": "hans", "format": "hdf5", "ident": 2},
        {"type": "remote", "format": "http", "ident": 3},
    ]
    sorted_list = sorted(bnlist, key=feat_basin.basin_priority_sorted_key)
    assert sorted_list[0]["ident"] == 1
    assert sorted_list[1]["ident"] == 3
    assert sorted_list[2]["ident"] == 0
    assert sorted_list[3]["ident"] == 2


@pytest.mark.parametrize("btype,bformat,sortval", [
    ["file", "hdf5", "aa"],
    ["remote", "http", "bb"],
    ["remote", "s3", "bc"],
    ["remote", "dcor", "bd"],
    ["peter", "hdf5", "za"],
    ["remote", "hans", "bz"],
    ["hans", "peter", "zz"],
]
                         )
def test_basin_sorting_key(btype, bformat, sortval):
    bdict = {"type": btype,
             "format": bformat,
             }
    assert feat_basin.basin_priority_sorted_key(bdict) == sortval


def test_basin_hierarchy_trace():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")
        # Next, store the basin information in the new dataset
        bdat = {
            "type": "file",
            "format": "hdf5",
            "features": ["trace"],
            "paths": [
                str(h5path),  # absolute path name
            ]
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = hw.h5file.require_group("basins")
        hw.write_text(basins, "invalid_format_basin", blines)

    ds = dclab.new_dataset(h5path_small)
    assert "trace" not in ds.features_innate
    ds2 = dclab.new_dataset(ds)
    assert "trace" in ds
    assert "trace" in ds2
    assert "fl1_raw" in ds["trace"]
    assert np.allclose(
        np.mean(ds2["trace"]["fl1_raw"][0]),
        24.5785536159601,
        atol=0, rtol=1e-5
    )


def test_basin_hierarchy_trace_missing():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")
        # Next, store the basin information in the new dataset
        bdat = {
            "type": "file",
            "format": "hdf5",
            "features": ["trace"],
            "paths": [
                str(h5path),  # absolute path name
            ]
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = hw.h5file.require_group("basins")
        hw.write_text(basins, "invalid_format_basin", blines)

    h5path.unlink()

    ds = dclab.new_dataset(h5path_small)
    ds2 = dclab.new_dataset(ds)
    ds2.apply_filter()
    assert "trace" not in ds2
    with pytest.raises(KeyError, match="does not contain the feature"):
        ds2["trace"]


def test_basin_unsupported_basin_format():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")
        # Next, store the basin information in the new dataset
        bdat = {
            "type": "file",
            "format": "peter",
            "paths": [
                "fake.rtdc",  # fake path
                str(h5path),  # absolute path name
            ]
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = hw.h5file.require_group("basins")
        hw.write_text(basins, "invalid_format_basin", blines)

    h5path.unlink()

    with pytest.warns(UserWarning,
                      match="Encountered unsupported basin format 'peter'"):
        with new_dataset(h5path_small) as ds:
            assert "image" not in ds
            assert not ds.features_basin


def test_basin_unsupported_basin_type():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")
        # Next, store the basin information in the new dataset
        bdat = {
            "type": "peter",
            "format": "hdf5",
            "paths": [
                "fake.rtdc",  # fake path
                str(h5path),  # absolute path name
            ]
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = hw.h5file.require_group("basins")
        hw.write_text(basins, "invalid_type_basin", blines)

    h5path.unlink()

    with pytest.warns(UserWarning,
                      match="Encountered unsupported basin type 'peter'"):
        with new_dataset(h5path_small) as ds:
            assert "image" not in ds
            assert not ds.features_basin
