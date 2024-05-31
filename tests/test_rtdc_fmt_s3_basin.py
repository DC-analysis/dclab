import json
import uuid

import h5py
import numpy as np

import pytest

from dclab import new_dataset, RTDCWriter
from dclab.rtdc_dataset.fmt_s3 import S3Basin, RTDC_S3
from dclab.rtdc_dataset.feat_basin import BasinNotAvailableError


from helper_methods import DCOR_AVAILABLE, retrieve_data


pytest.importorskip("boto3")

if not DCOR_AVAILABLE:
    pytest.skip("No connection to DCOR", allow_module_level=True)

s3_url = ("https://objectstore.hpccloud.mpcdf.mpg.de/"
          "circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
          "resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0")


def test_basin_as_dict(tmp_path):
    tmp_path = tmp_path.resolve()
    h5path = tmp_path / "test_basin_s3.rtdc"

    with h5py.File(h5path, "a") as dst, RTDC_S3(s3_url) as src:
        # Store non-existent basin information
        with RTDCWriter(dst, mode="append") as hw:
            meta = src.config.as_dict(pop_filtering=True)
            hw.store_metadata(meta)
            hw.store_basin(basin_name="example basin",
                           basin_type="remote",
                           basin_format="s3",
                           basin_locs=[s3_url],
                           basin_descr="an example S3 test basin",
                           )

    with new_dataset(h5path) as ds:
        assert ds._enable_basins
        bdict = ds.basins[0].as_dict()
        assert bdict["basin_name"] == "example basin"
        assert bdict["basin_type"] == "remote"
        assert bdict["basin_format"] == "s3"
        assert bdict["basin_locs"] == [s3_url]
        assert bdict["basin_descr"] == "an example S3 test basin"

    # Now use the data from `bdict` to create a new basin
    h5path_two = h5path.with_name("smaller_two.rtdc")

    # Dataset creation
    with RTDCWriter(h5path_two) as hw:
        # first, copy all the scalar features to the new file
        hw.store_metadata(meta)
        hw.store_basin(**bdict)

    with new_dataset(h5path_two) as ds2:
        bdict2 = ds2.basins[0].as_dict()
        assert bdict2["basin_name"] == "example basin"
        assert bdict2["basin_type"] == "remote"
        assert bdict2["basin_format"] == "s3"
        assert bdict2["basin_locs"] == [s3_url]
        assert bdict2["basin_descr"] == "an example S3 test basin"


@pytest.mark.parametrize("url", [
    "https://example.com/nonexistentbucket/nonexistentkey",
    f"https://objectstore.hpccloud.mpcdf.mpg.de/noexist-{uuid.uuid4()}/key",
])
def test_basin_not_available(url):
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")

    # Dataset creation
    with h5py.File(h5path, "a") as dst:
        # Store non-existent basin information
        bdat = {
            "type": "remote",
            "format": "s3",
            "urls": [
                # does not exist
                url
            ]
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = dst.require_group("basins")
        with RTDCWriter(dst, mode="append") as hw:
            hw.write_text(basins, "my_basin", blines)

    # Open the dataset and check whether basin is missing
    with new_dataset(h5path) as ds:
        assert not ds.features_basin
        # This is a very subtle test for checking whether invalid basins
        # are just ignored:
        _ = ds["index"]

    # Also test that on a lower level
    bn = S3Basin("https://example.com/nonexistentbucket/nonexistentkey")
    assert not bn.is_available()
    with pytest.raises(BasinNotAvailableError, match="is not available"):
        _ = bn.ds


def test_create_basin_file_non_matching_identifier(tmp_path):
    h5path = tmp_path / "test_basin_s3.rtdc"

    with h5py.File(h5path, "a") as dst, RTDC_S3(s3_url) as src:
        # Store non-existent basin information
        bdat = {
            "type": "remote",
            "format": "s3",
            "urls": [s3_url],
            "features": ["deform"],
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = dst.require_group("basins")
        with RTDCWriter(dst, mode="append") as hw:
            hw.write_text(basins, "my_basin", blines)
            meta = src.config.as_dict(pop_filtering=True)
            meta["experiment"]["run identifier"] = "hoolahoop"
            hw.store_metadata(meta)

    with new_dataset(h5path) as ds:
        assert ds.basins
        # The feature shows up as available...
        assert ds.features_basin == ["deform"]
        # ...but it is actually not, since the run identifier does not match
        # and therefore dclab does not allow the user to access it.
        with (pytest.warns(UserWarning, match="but I cannot get its data"),
              pytest.raises(KeyError, match="deform")):
            _ = ds["deform"]


def test_create_basin_file_with_no_data(tmp_path):
    h5path = tmp_path / "test_basin_s3.rtdc"

    with h5py.File(h5path, "a") as dst, RTDC_S3(s3_url) as src:
        # Store non-existent basin information
        bdat = {
            "type": "remote",
            "format": "s3",
            "urls": [s3_url]
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = dst.require_group("basins")
        with RTDCWriter(dst, mode="append") as hw:
            hw.write_text(basins, "my_basin", blines)
            meta = src.config.as_dict(pop_filtering=True)
            hw.store_metadata(meta)

    with new_dataset(h5path) as ds:
        assert ds.features_basin
        assert len(ds) == 5000
        assert np.allclose(ds["deform"][0], 0.009741939,
                           atol=0, rtol=1e-5)


def test_create_basin_file_with_one_feature(tmp_path):
    h5path = tmp_path / "test_basin_s3.rtdc"

    with h5py.File(h5path, "a") as dst, RTDC_S3(s3_url) as src:
        # Store non-existent basin information
        bdat = {
            "type": "remote",
            "format": "s3",
            "urls": [s3_url],
            "features": ["deform"],
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = dst.require_group("basins")
        with RTDCWriter(dst, mode="append") as hw:
            hw.write_text(basins, "my_basin", blines)
            meta = src.config.as_dict(pop_filtering=True)
            hw.store_metadata(meta)

    with new_dataset(h5path) as ds:
        assert ds.features_basin
        assert len(ds) == 5000
        assert "deform" in ds.features_basin
        assert "area_um" not in ds.features_basin
        assert "deform" in ds
        assert "area_um" not in ds
        assert np.allclose(ds["deform"][0], 0.009741939,
                           atol=0, rtol=1e-5)
