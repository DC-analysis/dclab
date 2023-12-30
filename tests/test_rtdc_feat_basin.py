import json

import h5py

import pytest

from dclab import new_dataset, rtdc_dataset, RTDCWriter


from helper_methods import retrieve_data


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
