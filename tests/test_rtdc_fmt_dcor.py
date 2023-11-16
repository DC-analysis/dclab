"""Test DCOR format"""
import pathlib
import socket

import dclab
from dclab.rtdc_dataset.fmt_dcor import RTDC_DCOR, is_dcor_url
import numpy as np
import pytest

from helper_methods import retrieve_data


pytest.importorskip("requests")
pytest.importorskip("fsspec")


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        s.connect(("dcor.mpl.mpg.de", 443))
    except (socket.gaierror, OSError):
        pytest.skip("No connection to DCOR",
                    allow_module_level=True)


class MockAPIHandler(dclab.rtdc_dataset.fmt_dcor.api.APIHandler):
    def __init__(self, *args, **kwargs):
        super(MockAPIHandler, self).__init__(*args, **kwargs)
        # We are mocking only API version 1
        self.dcserv_api_version = 1

    def get(self, query, feat=None, trace=None, event=None):
        """Mocks communication with the DCOR API"""
        h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
        with dclab.new_dataset(h5path) as ds:
            if query == "size":
                return len(ds)
            elif query == "basins":
                return []
            elif query == "metadata":
                return ds.config
            elif query == "feature_list":
                return ds.features
            elif query == "feature" and dclab.dfn.scalar_feature_exists(feat):
                return ds[feat]
            elif query == "feature" and feat == "trace":
                return ds["trace"][trace][event]
            elif query == "trace_list":
                return sorted(ds["trace"].keys())
            else:
                return ds[feat][event]


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_dcor_base(monkeypatch):
    monkeypatch.setattr(dclab.rtdc_dataset.fmt_dcor.api,
                        "APIHandler",
                        MockAPIHandler)
    with dclab.new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip")) as ds:
        dso = dclab.new_dataset(
            "https://example.com/api/3/action/dcserv?id="
            "b1404eb5-f661-4920-be79-5ff4e85915d1")
        assert len(dso) == len(ds)
        assert dso.config["setup"]["channel width"] == \
            ds.config["setup"]["channel width"]
        assert ds["area_um"].ndim == 1  # important for matplotlib
        assert np.all(dso["area_um"] == ds["area_um"])
        assert np.all(dso["area_um"] == ds["area_um"])  # test cache
        assert np.all(dso["image"][4] == ds["image"][4])
        assert len(dso["image"]) == len(ds)
        for key in dso._events:
            assert key in ds
        for m, n in zip(dso["mask"], ds["mask"]):
            assert np.all(m == n)
        # compute an ancillary feature
        assert np.all(dso["volume"] == ds["volume"])
        assert np.all(dso["volume"] == ds["volume"])  # test cache
        # trace
        assert sorted(dso["trace"].keys()) == sorted(ds["trace"].keys())
        assert len(dso["trace"]["fl1_raw"]) == len(ds["trace"]["fl1_raw"])
        assert np.all(dso["trace"]["fl1_raw"][1] == ds["trace"]["fl1_raw"][1])
        for t1, t2 in zip(dso["trace"]["fl1_raw"], ds["trace"]["fl1_raw"]):
            assert np.all(t1 == t2)


def test_dcor_cache_scalar():
    # Testing the cache is only relevant for api version 1
    # calibration beads
    with dclab.new_dataset("fb719fb2-bd9f-817a-7d70-f4002af916f0",
                           dcserv_api_version=1) as ds:
        # sanity checks
        assert len(ds) == 5000
        assert "area_um" in ds

        area_um = ds["area_um"]
        assert ds["area_um"] is area_um, "Check proper caching"
        # provoke cache deletion
        ds._events._scalar_cache.pop("area_um")
        assert ds["area_um"] is not area_um, "test removal from cache"


def test_dcor_cache_trace():
    # Testing the cache is only relevant for api version 1
    # calibration beads
    with dclab.new_dataset("fb719fb2-bd9f-817a-7d70-f4002af916f0",
                           dcserv_api_version=1) as ds:
        # sanity checks
        assert len(ds) == 5000
        assert "trace" in ds

        trace0 = ds["trace"]["fl1_raw"][0]
        assert ds["trace"]["fl1_raw"][0] is trace0, "Check proper caching"
        assert ds["trace"]["fl1_raw"][1] is not trace0, "Check proper caching"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_dcor_data():
    # reticulocytes.rtdc contains contour data
    with dclab.new_dataset("13247dd0-3d8b-711d-a410-468b4de6fb7a") as ds:
        assert np.allclose(ds["circ"][0],
                           0.7309052348136902,
                           rtol=0, atol=1e-5)
        assert np.allclose(ds["area_um"][391], 37.5122, rtol=0, atol=1e-5)
        assert np.all(ds["contour"][24][22] == np.array([87, 61]))
        assert np.median(ds["image"][1]) == 58
        assert np.sum(ds["mask"][11]) == 332
        assert np.sum(ds["mask"][11]) == 332
        assert np.median(ds["trace"]["fl1_raw"][200]) == 183.0
        assert np.sum(ds["trace"]["fl1_median"][2167]) == 183045


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_dcor_hash():
    with dclab.new_dataset("fb719fb2-bd9f-817a-7d70-f4002af916f0") as ds:
        # hash includes the full URL (path)
        assert ds.hash == "7250277b41b757cbe09647a58e8ca4ce"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_dcor_hierarchy(monkeypatch):
    monkeypatch.setattr(dclab.rtdc_dataset.fmt_dcor.api,
                        "APIHandler",
                        MockAPIHandler)
    dso = dclab.new_dataset("https://example.com/api/3/action/dcserv?id="
                            "b1404eb5-f661-4920-be79-5ff4e85915d5")
    dsh = dclab.new_dataset(dso)
    assert np.all(dso["area_um"] == dsh["area_um"])


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_dcor_logs():
    with dclab.new_dataset("fb719fb2-bd9f-817a-7d70-f4002af916f0") as ds:
        assert len(ds.logs) >= 2  # there might be others
        assert ds.logs["log"][0] \
               == "[LOG] number of written datasets 0  10:04:05.893"
        assert "dclab-condense" in ds.logs


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_dcor_shape_contour():
    # calibration beads
    with dclab.new_dataset("fb719fb2-bd9f-817a-7d70-f4002af916f0") as ds:
        assert len(ds["contour"]) == 5000
        assert ds["contour"].shape == (5000, np.nan, 2)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_dcor_shape_image():
    # calibration beads
    with dclab.new_dataset("fb719fb2-bd9f-817a-7d70-f4002af916f0") as ds:
        assert len(ds["image"]) == 5000
        assert ds["image"].shape == (5000, 80, 250)
        assert ds["image"][0].shape == (80, 250)
        assert ds["image"][0][0].shape == (250,)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_dcor_shape_mask():
    # calibration beads
    with dclab.new_dataset("fb719fb2-bd9f-817a-7d70-f4002af916f0") as ds:
        assert len(ds["mask"]) == 5000
        assert ds["mask"].shape == (5000, 80, 250)
        assert ds["mask"][0].shape == (80, 250)
        assert ds["mask"][0][0].shape == (250,)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_dcor_shape_trace():
    # calibration beads
    with dclab.new_dataset("fb719fb2-bd9f-817a-7d70-f4002af916f0") as ds:
        assert len(ds["trace"]) == 6
        assert ds["trace"].shape == (6, 5000, 177)
        assert len(ds["trace"]["fl1_raw"]) == 5000
        assert ds["trace"]["fl1_raw"].shape == (5000, 177)
        assert len(ds["trace"]["fl1_raw"][0]) == 177
        assert len(ds["trace"]["fl1_raw"][0]) != 5000
        assert ds["trace"]["fl1_raw"][0].shape == (177,)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.parametrize("idxs", [slice(0, 5, 2),
                                  np.array([0, 2, 4]),
                                  [0, 2, 4]
                                  ])
def test_dcor_slicing_contour(idxs):
    """Test slicing of contour data"""
    # reticulocytes.rtdc contains contour data
    with dclab.new_dataset("13247dd0-3d8b-711d-a410-468b4de6fb7a") as ds:

        data_ref = [
            ds["contour"][0],
            ds["contour"][2],
            ds["contour"][4],
        ]

        data_sliced = ds["contour"][idxs]

        assert np.all(data_sliced[0] == data_ref[0])
        assert np.all(data_sliced[1] == data_ref[1])
        assert np.all(data_sliced[2] == data_ref[2])


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.parametrize("feat", ["image", "mask"])
@pytest.mark.parametrize("idxs", [slice(0, 5, 2),
                                  np.array([0, 2, 4]),
                                  [0, 2, 4]
                                  ])
def test_dcor_slicing_image_mask(feat, idxs):
    """Test slicing of image/mask data"""
    with dclab.new_dataset("fb719fb2-bd9f-817a-7d70-f4002af916f0") as ds:
        data_ref = [
            ds[feat][0],
            ds[feat][2],
            ds[feat][4],
        ]

        data_sliced = ds[feat][idxs]

        assert np.all(data_sliced[0] == data_ref[0])
        assert np.all(data_sliced[1] == data_ref[1])
        assert np.all(data_sliced[2] == data_ref[2])


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.parametrize("idxs", [slice(0, 5, 2),
                                  np.array([0, 2, 4]),
                                  [0, 2, 4]
                                  ])
def test_dcor_slicing_trace(idxs):
    """Test slicing of trace data"""
    with dclab.new_dataset("fb719fb2-bd9f-817a-7d70-f4002af916f0") as ds:
        data_ref = [
            ds["trace"]["fl1_raw"][0],
            ds["trace"]["fl1_raw"][2],
            ds["trace"]["fl1_raw"][4],
        ]

        data_sliced = ds["trace"]["fl1_raw"][idxs]

        assert np.all(data_sliced[0] == data_ref[0])
        assert np.all(data_sliced[1] == data_ref[1])
        assert np.all(data_sliced[2] == data_ref[2])


@pytest.mark.parametrize("target,kwargs", [
    # HTTPS
    ("https://example.com/api/3/action/dcserv?"
     "id=b1404eb5-f661-4920-be79-5ff4e85915d5",
     {"url": "b1404eb5-f661-4920-be79-5ff4e85915d5",
      "use_ssl": True,
      "host": "example.com"}
     ),
    ("https://example.com/api/3/action/dcserv?"
     "id=b1404eb5-f661-4920-be79-5ff4e85915d5",
     {"url": "example.com/api/3/action/dcserv?id="
             "b1404eb5-f661-4920-be79-5ff4e85915d5",
      "use_ssl": True,
      "host": "example.com"}
     ),
    ("https://example.com/api/3/action/dcserv?"
     "id=b1404eb5-f661-4920-be79-5ff4e85915d5",
     {"url": "example.com/api/3/action/dcserv?id="
             "b1404eb5-f661-4920-be79-5ff4e85915d5",
      "use_ssl": None,
      "host": "example.com"}
     ),
    ("https://example.com/api/3/action/dcserv?"
     "id=b1404eb5-f661-4920-be79-5ff4e85915d5",
     {"url": "b1404eb5-f661-4920-be79-5ff4e85915d5",
      "use_ssl": None,
      "host": "example.com"}
     ),
    ("https://example.com/api/3/action/dcserv?"
     "id=b1404eb5-f661-4920-be79-5ff4e85915d5",
     {"url": "b1404eb5-f661-4920-be79-5ff4e85915d5",
      "use_ssl": None,
      # sneak in a scheme into host
      "host": "https://example.com"}
     ),
    ("https://example.com/api/3/action/dcserv?"
     "id=b1404eb5-f661-4920-be79-5ff4e85915d5",
     {"url": "https://example.com/api/3/action/dcserv?id="
             "b1404eb5-f661-4920-be79-5ff4e85915d5",
      "use_ssl": None,
      # `host` does not override things
      "host": "example2.com"}
     ),
    # HTTP
    ("http://example.com/api/3/action/dcserv?"
     "id=b1404eb5-f661-4920-be79-5ff4e85915d5",
     {"url": "example.com/api/3/action/dcserv?id="
             "b1404eb5-f661-4920-be79-5ff4e85915d5",
      "use_ssl": False,
      "host": "example.com"}
     ),
    ("http://example.com/api/3/action/dcserv?"
     "id=b1404eb5-f661-4920-be79-5ff4e85915d5",
     {"url": "https://example.com/api/3/action/dcserv?id="
             "b1404eb5-f661-4920-be79-5ff4e85915d5",
      "use_ssl": False,
      "host": "example.com"}
     ),
    ("http://example.com/api/3/action/dcserv?"
     "id=b1404eb5-f661-4920-be79-5ff4e85915d5",
     {"url": "b1404eb5-f661-4920-be79-5ff4e85915d5",
      "use_ssl": False,
      "host": "example.com"}
     ),
    ("http://example.com/api/3/action/dcserv?"
     "id=b1404eb5-f661-4920-be79-5ff4e85915d5",
     {"url": "http://example.com/api/3/action/dcserv?id="
             "b1404eb5-f661-4920-be79-5ff4e85915d5",
      "use_ssl": None,
      "host": "example.com"}
     ),
    ("http://example.com/api/3/action/dcserv?"
     "id=b1404eb5-f661-4920-be79-5ff4e85915d5",
     {"url": "b1404eb5-f661-4920-be79-5ff4e85915d5",
      "use_ssl": False,
      # sneak in a scheme into host
      "host": "https://example.com"}
     ),
    ("http://example.com/api/3/action/dcserv?"
     "id=b1404eb5-f661-4920-be79-5ff4e85915d5",
     {"url": "http://example.com/api/3/action/dcserv?id="
             "b1404eb5-f661-4920-be79-5ff4e85915d5",
      "use_ssl": False,
      # `host` does not override things
      "host": "example2.com"}
     ),
])
def test_get_full_url(target, kwargs):
    assert target == RTDC_DCOR.get_full_url(**kwargs)


def test_is_dcor_url():
    assert is_dcor_url("2cea205f-2d9d-26d0-b44c-0a11d5379152")
    assert not is_dcor_url("2cea205f-2d9d")
    assert is_dcor_url("https://example.com/api/3/action/dcserv?id="
                       "2cea205f-2d9d-26d0-b44c-0a11d5379152")
    assert is_dcor_url("http://example.com/api/3/action/dcserv?id="
                       "2cea205f-2d9d-26d0-b44c-0a11d5379152")
    assert is_dcor_url("example.com/api/3/action/dcserv?id="
                       "2cea205f-2d9d-26d0-b44c-0a11d5379152")
    assert not is_dcor_url(
        pathlib.Path("example.com/api/3/action/dcserv?id="
                     "2cea205f-2d9d-26d0-b44c-0a11d5379152"))
    assert not is_dcor_url(2.0)
    assert not is_dcor_url("/home/peter/pan")
    assert not is_dcor_url("https://example.com/api/3/action/dcserv?id="
                           "2cea205f-2d9d-26d0-b44c-")


def test_load_nonexistent_file_issue81():
    """https://github.com/DC-analysis/dclab/issues/81"""
    try:
        dclab.new_dataset("path/does/not/exist.rtdc")
    except FileNotFoundError:
        pass
    else:
        assert False, "Non-existent files should raise FileNotFoundError"
