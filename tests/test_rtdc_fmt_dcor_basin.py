import json
import time
import uuid

import h5py
import numpy as np

import pytest

from dclab import new_dataset, RTDCWriter
from dclab.rtdc_dataset.fmt_dcor import DCORBasin, RTDC_DCOR
from dclab.rtdc_dataset.feat_basin import BasinNotAvailableError

from helper_methods import DCOR_AVAILABLE, retrieve_data


pytest.importorskip("requests")

if not DCOR_AVAILABLE:
    pytest.skip("No connection to DCOR", allow_module_level=True)

dcor_url = ("https://dcor.mpl.mpg.de/api/3/action/dcserv?id="
            "57ecde5d-f896-4599-ba35-d1be7defc6fe")


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_basin_as_dict(tmp_path):
    tmp_path = tmp_path.resolve()
    h5path = tmp_path / "test_basin_dcor.rtdc"

    with h5py.File(h5path, "a") as dst, RTDC_DCOR(dcor_url) as src:
        # Store non-existent basin information
        with RTDCWriter(dst, mode="append") as hw:
            meta = src.config.as_dict(pop_filtering=True)
            hw.store_metadata(meta)
            hw.store_basin(basin_name="example basin",
                           basin_type="remote",
                           basin_format="dcor",
                           basin_locs=[dcor_url],
                           basin_descr="an example DCOR test basin",
                           )

    with new_dataset(h5path) as ds:
        assert ds._enable_basins
        bdict = ds.basins[0].as_dict()
        assert bdict["basin_name"] == "example basin"
        assert bdict["basin_type"] == "remote"
        assert bdict["basin_format"] == "dcor"
        assert bdict["basin_locs"] == [dcor_url]
        assert bdict["basin_descr"] == "an example DCOR test basin"

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
        assert bdict2["basin_format"] == "dcor"
        assert bdict2["basin_locs"] == [dcor_url]
        assert bdict2["basin_descr"] == "an example DCOR test basin"


def test_basins_basins_get_dicts_update():
    with RTDC_DCOR(dcor_url) as ds:
        basin_dict1 = ds.basins_get_dicts()
        basin_dict2 = ds.basins_get_dicts()
        assert basin_dict1 is basin_dict2, "basin dict should be cached"
        ds.cache_basin_dict_time = 0.1
        time.sleep(0.2)
        basin_dict3 = ds.basins_get_dicts()
        assert basin_dict3 is not basin_dict1, "cache should be invalidated"


@pytest.mark.parametrize("url", [
    "https://example.com/nonexistentbucket/nonexistentkey",
    f"https://objectstore.hpccloud.mpcdf.mpg.de/noexist-{uuid.uuid4()}/key",
])
@pytest.mark.filterwarnings(
    "ignore::dclab.http_utils.ConnectionTimeoutWarning")
def test_basin_not_available(url):
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")

    # Dataset creation
    with h5py.File(h5path, "a") as dst:
        # Store non-existent basin information
        bdat = {
            "type": "remote",
            "format": "dcor",
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
    bn = DCORBasin("https://dcor.mpl.mpg.de/api/3/action/dcserv?id="
                   "00000000-0000-0000-0000-000000000000")
    assert not bn.is_available()
    with pytest.raises(BasinNotAvailableError, match="is not available"):
        _ = bn.ds


def test_create_basin_file_non_matching_identifier(tmp_path):
    h5path = tmp_path / "test_basin_dcor.rtdc"

    with h5py.File(h5path, "a") as dst, RTDC_DCOR(dcor_url) as src:
        # Store non-existent basin information
        bdat = {
            "type": "remote",
            "format": "dcor",
            "urls": [dcor_url],
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
        #
        # Until a workaround is found for invalid basin URLs that return a
        # status code of 200, do this test which should raise a warning,
        # because `__contains__` returns True for "trace", but the trace data
        # are nowhere to find.
        with (pytest.warns(UserWarning, match="but I cannot get its data"),
              pytest.raises(KeyError, match="deform")):
            _ = ds["deform"]


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_create_basin_file_with_no_data(tmp_path):
    h5path = tmp_path / "test_basin_dcor.rtdc"

    with h5py.File(h5path, "a") as dst, RTDC_DCOR(dcor_url) as src:
        # Store non-existent basin information
        bdat = {
            "type": "remote",
            "format": "dcor",
            "urls": [dcor_url]
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = dst.require_group("basins")
        with RTDCWriter(dst, mode="append") as hw:
            hw.write_text(basins, "my_basin", blines)
            meta = src.config.as_dict(pop_filtering=True)
            hw.store_metadata(meta)

    with new_dataset(h5path) as ds:
        # This is essentially a nested basin features test. The basin is
        # a DCOR dataset which has two basins, the condensed version of the
        # data and the full version of the data as HTTP basins.
        assert len(ds.basins) == 1
        bn = ds.basins[0]
        assert len(bn.ds.basins) == 2
        assert ds.features_basin
        assert len(ds) == 28
        assert np.allclose(ds["deform"][0], 0.05335504858810891,
                           atol=0, rtol=1e-5)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_create_basin_file_with_one_feature(tmp_path):
    h5path = tmp_path / "test_basin_dcor.rtdc"

    with h5py.File(h5path, "a") as dst, RTDC_DCOR(dcor_url) as src:
        # Store non-existent basin information
        bdat = {
            "type": "remote",
            "format": "dcor",
            "urls": [dcor_url],
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
        assert len(ds) == 28
        assert "deform" in ds.features_basin
        assert "area_um" not in ds.features_basin
        assert "deform" in ds
        assert "area_um" not in ds
        assert np.allclose(ds["deform"][0], 0.05335504858810891,
                           atol=0, rtol=1e-5)
