import json
import uuid
import socket

import h5py
import numpy as np

import pytest

from dclab import new_dataset, RTDCWriter
from dclab.rtdc_dataset.fmt_s3 import S3Basin


from helper_methods import retrieve_data


pytest.importorskip("s3fs")


s3_url = ("https://objectstore.hpccloud.mpcdf.mpg.de/"
          "circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
          "resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0")


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        s.connect(("dcor.mpl.mpg.de", 443))
    except (socket.gaierror, OSError):
        pytest.skip("No connection to DCOR",
                    allow_module_level=True)


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

    # Also test that on a lower level
    bn = S3Basin("https://example.com/nonexistentbucket/nonexistentkey")
    assert not bn.is_available()
    with pytest.raises(ValueError, match="is not available"):
        _ = bn.ds


def test_create_basin_file_with_no_data(tmp_path):
    h5path = tmp_path / "test_basin_s3.rtdc"

    with h5py.File(h5path, "a") as dst, new_dataset(s3_url) as src:
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
            meta = dict(src.config)
            meta.pop("filtering")
            hw.store_metadata(meta)

    with new_dataset(h5path) as ds:
        assert ds.features_basin
        assert len(ds) == 5000
        assert np.allclose(ds["deform"][0], 0.009741939,
                           atol=0, rtol=1e-5)
