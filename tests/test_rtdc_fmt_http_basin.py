import json
import uuid

import h5py
import numpy as np

import pytest

import dclab
from dclab.http_utils import ETagNotInResponseHeaderWarning
from dclab import new_dataset, RTDCWriter
from dclab.rtdc_dataset.fmt_http import HTTPBasin, RTDC_HTTP
from dclab.rtdc_dataset.feat_basin import BasinNotAvailableError


from helper_methods import DCOR_AVAILABLE, retrieve_data


pytest.importorskip("requests")

if not DCOR_AVAILABLE:
    pytest.skip("No connection to DCOR", allow_module_level=True)


# 250209_Blood_2025-02-09_09.46_M003_Reference_dcn_export_28.rtdc
http_url = ("https://objectstore.hpccloud.mpcdf.mpg.de/"
            "circle-442e6d53-c48b-46eb-873c-4a0f98f3827d/"
            "resource/57e/cde/5d-f896-4599-ba35-d1be7defc6fe")

# calibration beads
http_url_trace = ("https://objectstore.hpccloud.mpcdf.mpg.de/"
                  "circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
                  "resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0")


def test_basin_as_dict(tmp_path):
    tmp_path = tmp_path.resolve()
    h5path = tmp_path / "test_basin_http.rtdc"

    with h5py.File(h5path, "a") as dst, RTDC_HTTP(http_url) as src:
        # Store non-existent basin information
        with RTDCWriter(dst, mode="append") as hw:
            meta = src.config.as_dict(pop_filtering=True)
            hw.store_metadata(meta)
            hw.store_basin(basin_name="example basin",
                           basin_type="remote",
                           basin_format="http",
                           basin_locs=[http_url],
                           basin_descr="an example http test basin",
                           )

    with new_dataset(h5path) as ds:
        assert ds._enable_basins
        bdict = ds.basins[0].as_dict()
        assert bdict["basin_name"] == "example basin"
        assert bdict["basin_type"] == "remote"
        assert bdict["basin_format"] == "http"
        assert bdict["basin_locs"] == [http_url]
        assert bdict["basin_descr"] == "an example http test basin"

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
        assert bdict2["basin_format"] == "http"
        assert bdict2["basin_locs"] == [http_url]
        assert bdict2["basin_descr"] == "an example http test basin"


def test_basin_as_dict_netloc_vs_hostname(tmp_path):
    tmp_path = tmp_path.resolve()
    h5path = tmp_path / "test_basin_http.rtdc"

    # note the port is included here
    http_url_netloc = ("https://objectstore.hpccloud.mpcdf.mpg.de:443/"
                       "circle-442e6d53-c48b-46eb-873c-4a0f98f3827d/"
                       "resource/57e/cde/5d-f896-4599-ba35-d1be7defc6fe")

    with h5py.File(h5path, "a") as dst, RTDC_HTTP(http_url) as src:
        # Store non-existent basin information
        with RTDCWriter(dst, mode="append") as hw:
            meta = src.config.as_dict(pop_filtering=True)
            hw.store_metadata(meta)
            hw.store_basin(basin_name="example basin",
                           basin_type="remote",
                           basin_format="http",
                           basin_locs=[http_url_netloc],
                           basin_descr="an example http test basin",
                           )

    with new_dataset(h5path) as ds:
        assert len(ds) == 28
        # This failed in <0.56.0, because `netloc` was used instead of
        # `hostname` when connecting to the socket to check whether the
        # server is available.
        assert np.allclose(ds["deform"][0], 0.05335504858810891,
                           rtol=0, atol=1e-7)


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
            "format": "http",
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
    bn = HTTPBasin("https://example.com/nonexistentbucket/nonexistentkey")
    assert not bn.is_available()
    with pytest.raises(BasinNotAvailableError, match="is not available"):
        _ = bn.ds


def test_create_basin_file_non_matching_identifier(tmp_path):
    h5path = tmp_path / "test_basin_http.rtdc"

    with h5py.File(h5path, "a") as dst, RTDC_HTTP(http_url) as src:
        # Store non-existent basin information
        bdat = {
            "type": "remote",
            "format": "http",
            "urls": [http_url],
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
    h5path = tmp_path / "test_basin_http.rtdc"

    with h5py.File(h5path, "a") as dst, RTDC_HTTP(http_url) as src:
        # Store non-existent basin information
        bdat = {
            "type": "remote",
            "format": "http",
            "urls": [http_url]
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
        assert np.allclose(ds["deform"][0], 0.05335504858810891,
                           atol=0, rtol=1e-5)


def test_create_basin_file_with_one_feature(tmp_path):
    h5path = tmp_path / "test_basin_http.rtdc"

    with h5py.File(h5path, "a") as dst, RTDC_HTTP(http_url) as src:
        # Store non-existent basin information
        bdat = {
            "type": "remote",
            "format": "http",
            "urls": [http_url],
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


def test_trace_availability(tmp_path):
    h5path = tmp_path / "test_basin_http.rtdc"

    with h5py.File(h5path, "a") as dst, RTDC_HTTP(http_url_trace) as src:
        # Store non-existent basin information
        bdat = {
            "type": "remote",
            "format": "http",
            "urls": [http_url_trace],
            "features": ["trace"],
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = dst.require_group("basins")
        with RTDCWriter(dst, mode="append") as hw:
            hw.write_text(basins, "my_basin", blines)
            meta = src.config.as_dict(pop_filtering=True)
            hw.store_metadata(meta)

    with dclab.new_dataset(h5path) as ds:
        ds.filter.manual[0] = False
        with dclab.new_dataset(ds) as ds2:
            assert "trace" in ds2


def test_trace_availability_invalid(tmp_path):
    h5path = tmp_path / "test_basin_http.rtdc"

    with h5py.File(h5path, "a") as dst, RTDC_HTTP(http_url_trace) as src:
        # Store non-existent basin information
        bdat = {
            "type": "remote",
            "format": "http",
            "urls": ["https://dcor.mpl.mpg.de/api/3/action/status_show"],
            "features": ["trace"],
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = dst.require_group("basins")
        with RTDCWriter(dst, mode="append") as hw:
            hw.write_text(basins, "my_basin", blines)
            meta = src.config.as_dict(pop_filtering=True)
            hw.store_metadata(meta)

    with dclab.new_dataset(h5path) as ds:
        ds.filter.manual[:] = False
        ds.filter.manual[:2] = True
        ds.apply_filter()
        assert "trace" in ds
        # Until a workaround is found for invalid basin URLs that return a
        # status code of 200, do this test which should raise a warning,
        # because `__contains__` returns True for "trace", but the trace data
        # are nowhere to find.
        with (pytest.warns(UserWarning, match="but I cannot get its data"),
              pytest.warns(ETagNotInResponseHeaderWarning,
                           match="Got empty ETag header"),
              pytest.raises(KeyError, match="trace")):
            _ = ds["trace"]
        with (pytest.warns(UserWarning, match="but I cannot get its data"),
              pytest.warns(ETagNotInResponseHeaderWarning,
                           match="Got empty ETag header"),
              pytest.raises(KeyError, match="trace")):
            dclab.new_dataset(ds)
