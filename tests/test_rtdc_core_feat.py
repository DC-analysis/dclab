import pytest

import dclab
from dclab import RTDCWriter
from dclab.rtdc_dataset import rtdc_copy
from dclab.rtdc_dataset.fmt_http import RTDC_HTTP
import h5py

from helper_methods import DCOR_AVAILABLE, retrieve_data


http_url = ("https://objectstore.hpccloud.mpcdf.mpg.de/"
            "circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
            "resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0")


def test_features_local_basic():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    with dclab.new_dataset(h5path) as ds:
        # access time and index so that they are in the local features
        assert ds["index"][0] == 1
        assert len(ds["time"]) == 10
        assert ds.features_local == ds.features_loaded


def test_features_local_basin():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("small.rtdc")

    with h5py.File(h5path, "a") as h5:
        del h5["events/deform"]

    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_copy(src_h5file=src,
                  dst_h5file=hw.h5file,
                  features=["area_um"])
        hw.store_basin(basin_name="example basin",
                       basin_type="file",
                       basin_format="hdf5",
                       basin_locs=[h5path],
                       basin_descr="an example test basin",
                       )

    with dclab.new_dataset(h5path_small) as ds:
        # access time and index so that they are in the local features
        assert "area_um" in ds.features_local
        assert "area_msd" in ds.features_local
        # not accessed, thus not locally available
        assert "time" not in ds.features_local
        assert "time" in ds.features_loaded
        # area_um was removed above, but can be computed (ancillary)
        assert "deform" in ds.features_loaded
        assert "deform" not in ds.features_local
        assert "circ" in ds.features_local
        assert ds["deform"] is not None  # just access the feature
        assert "deform" in ds.features_local


@pytest.mark.skipif(not DCOR_AVAILABLE, reason="DCOR not accessible")
def test_features_local_remote():
    """Open a remote dataset and see whether local features are empty"""
    pytest.importorskip("requests")
    with RTDC_HTTP(http_url) as ds:
        assert not ds.features_local
        assert ds.features_loaded
        assert ds["deform"] is not None  # access a feature
        assert ds.features_local == ["deform"]


@pytest.mark.skipif(not DCOR_AVAILABLE, reason="DCOR not accessible")
def test_features_local_remote_basin(tmp_path):
    pytest.importorskip("requests")
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

    with dclab.new_dataset(h5path) as ds:
        assert not ds.features_local
        assert ds.features_loaded
        assert ds["deform"] is not None  # access a feature
        # The "circ" feature should not be downloaded for accessing the
        # "deform" feature.
        assert ds.features_local == ["deform"]
        assert "deform" not in ds.features_innate
        assert "deform" in ds.features_basin
        assert ds.basins[0].ds.features_local == ["deform"]
