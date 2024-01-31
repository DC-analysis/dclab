import numbers
from os.path import join
import tempfile

import h5py
import numpy as np

import pytest

import dclab
from dclab.rtdc_dataset import RTDCWriter, new_dataset
from dclab.rtdc_dataset.feat_temp import (
    register_temporary_feature, deregister_temporary_feature)

from helper_methods import retrieve_data, DCOR_AVAILABLE, BOTO3_AVAILABLE


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_basin_file():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_test = path.parent / "test.h5"
    # We basically create a file that consists only of the metadata.
    with RTDCWriter(path_test) as hw, new_dataset(path) as dsorig:
        hw.store_basin(basin_name="get-out",
                       basin_type="file",
                       basin_format="hdf5",
                       basin_locs=[path],
                       basin_descr="My very first basin-only dataset",
                       verify=True,
                       )
        meta = dsorig.config.as_dict(pop_filtering=True)
        hw.store_metadata(meta)

    # OK, now open the dataset and make sure that it contains all information.
    with new_dataset(path_test) as ds:
        assert len(ds) == 5
        assert ds.config["experiment"]["sample"] == "background image example"
        assert np.allclose(ds["deform"][2],
                           0.029980043,
                           atol=0,
                           rtol=1e-5)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_basin_feature_restriction():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_test = path.parent / "test.h5"
    # We basically create a file that consists only of the metadata.
    with RTDCWriter(path_test) as hw, new_dataset(path) as dsorig:
        hw.store_basin(basin_name="get-out",
                       basin_type="file",
                       basin_format="hdf5",
                       basin_locs=[path],
                       basin_descr="My very first basin-only dataset",
                       basin_feats=["deform"],
                       verify=True,
                       )
        meta = dsorig.config.as_dict(pop_filtering=True)
        hw.store_metadata(meta)
        # sanity check
        assert "area_um" in dsorig

    # Make sure *only* derorm is in the basin
    with dclab.new_dataset(path_test) as ds:
        assert "deform" in ds.features
        assert "area_um" not in ds.features
        assert "deform" in ds.features_basin


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_basin_feature_restriction_verify():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_test = path.parent / "test.h5"
    # We basically create a file that consists only of the metadata.
    with RTDCWriter(path_test) as hw:
        with pytest.raises(ValueError, match="blablabla"):
            hw.store_basin(basin_name="get-out",
                           basin_type="file",
                           basin_format="hdf5",
                           basin_locs=[path],
                           basin_descr="My very first basin-only dataset",
                           basin_feats=["blablabla"],
                           verify=True,
                           )


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_basin_file_relative():
    """Test whether storing the relative path works"""
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    path_test = path.parent / "test.h5"
    # We basically create a file that consists only of the metadata.
    with RTDCWriter(path_test) as hw, new_dataset(path) as dsorig:
        hw.store_basin(basin_name="get-out",
                       basin_type="file",
                       basin_format="hdf5",
                       basin_locs=[path],
                       basin_descr="My very first basin-only dataset",
                       verify=True
                       )
        meta = dsorig.config.as_dict(pop_filtering=True)
        hw.store_metadata(meta)

    # Now to the relative testing part. We just move everything one directory
    # down. So the absolute paths will not be valid anymore.
    newdir = path.parent / "new"
    newdir.mkdir()
    new_path = newdir / path.name
    new_path_test = newdir / path_test.name
    path.rename(new_path)
    path_test.rename(new_path_test)

    # OK, now open the dataset and make sure that it contains all information.
    with new_dataset(new_path_test) as ds:
        assert len(ds.basins) == 1
        assert len(ds) == 5
        assert ds.config["experiment"]["sample"] == "background image example"
        assert np.allclose(ds["deform"][2],
                           0.029980043,
                           atol=0,
                           rtol=1e-5)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_basin_file_identifier_mismatch():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    with RTDCWriter(path) as ho:
        ho.store_metadata({"experiment": {"run identifier": "captain-proton"}})

    path_test = path.parent / "test.h5"
    # We basically create a file that consists only of the metadata.
    with RTDCWriter(path_test) as hw, new_dataset(path) as dsorig:
        meta = dsorig.config.as_dict(pop_filtering=True)
        hw.store_metadata(meta)
        hw.store_metadata({"experiment": {"run identifier": "janeway"}})
        with pytest.raises(ValueError, match="identifier mismatch"):
            hw.store_basin(basin_name="get-out",
                           basin_type="file",
                           basin_format="hdf5",
                           basin_locs=[path],
                           basin_descr="My very first basin-only dataset",
                           verify=True
                           )
        # But it should work when verify is False
        hw.store_basin(basin_name="get-out",
                       basin_type="file",
                       basin_format="hdf5",
                       basin_locs=[path],
                       basin_descr="My very first basin-only dataset",
                       verify=False
                       )

    # Then, we have zero basins...
    with new_dataset(path_test) as ds:
        assert len(ds.basins) == 0

    # ...unless we change the run identifier to match the basin.
    with RTDCWriter(path_test) as hw:
        hw.store_metadata({"experiment": {"run identifier": "captain-proton"}})

    with new_dataset(path_test) as ds:
        assert len(ds.basins) == 1


@pytest.mark.skipif(not DCOR_AVAILABLE, reason="DCOR not reachable!")
@pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not available!")
def test_basin_url(tmp_path):
    # Create an empty dataset that links to an S3 instance
    # This is the calibration beads measurement.
    # https://dcor.mpl.mpg.de/dataset/figshare-7771184-v2/
    # resource/fb719fb2-bd9f-817a-7d70-f4002af916f0
    s3_url = ("https://objectstore.hpccloud.mpcdf.mpg.de/"
              "circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
              "resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0")
    with new_dataset(s3_url) as ds:
        meta = ds.config.as_dict(pop_filtering=True)

    path = tmp_path / "basins3test.rtdc"
    with RTDCWriter(path) as hw:
        hw.store_metadata(meta)
        hw.store_basin(basin_name="s3test",
                       basin_type="remote",
                       basin_format="s3",
                       basin_locs=[s3_url],
                       basin_descr="Me very first online dataset",
                       verify=True,
                       )

    with new_dataset(path) as ds:
        assert ds.basins[0].name == "s3test"
        assert ds.basins[0].basin_type == "remote"
        assert ds.basins[0].basin_format == "s3"
        assert ds.basins[0].location == s3_url
        assert ds.basins[0].description == "Me very first online dataset"
        assert len(ds) == 5000
        assert np.allclose(ds["deform"][0],
                           0.009741939,
                           atol=0,
                           rtol=1e-5)


def test_bulk_scalar():
    data = {"area_um": np.linspace(100.7, 110.9, 100)}
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_bulk_scalar_")
    with RTDCWriter(rtdc_file) as hw:
        for feat in data:
            hw.store_feature(feat, data[feat])
    # Read the file:
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        events = rtdc_data["events"]
        assert "area_um" in events.keys()
        assert np.all(events["area_um"][:] == data["area_um"])


def test_bulk_contour():
    num = 7
    contour = []
    for ii in range(5, num + 5):
        cii = np.arange(2 * ii).reshape(2, ii)
        contour.append(cii)
    data = {"area_um": np.linspace(100.7, 110.9, num),
            "contour": contour}
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_bulk_contour_")
    with RTDCWriter(rtdc_file) as hw:
        for feat in data:
            hw.store_feature(feat, data[feat])
    # Read the file:
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        events = rtdc_data["events"]
        assert "contour" in events.keys()
        assert np.allclose(events["contour"]["6"], contour[6])
        assert events["contour"]["1"].shape == (2, 6)


def test_bulk_image():
    num = 7
    image = np.zeros((20, 90, 50))
    image += np.arange(90).reshape(1, 90, 1)
    data = {"area_um": np.linspace(100.7, 110.9, num),
            "image": image}
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_bulk_image_")
    with RTDCWriter(rtdc_file) as hw:
        for feat in data:
            hw.store_feature(feat, data[feat])
    # Read the file:
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        events = rtdc_data["events"]
        assert "image" in events.keys()
        assert np.allclose(events["image"][6], image[6])


@pytest.mark.parametrize(
    "feat,dtype", [
        ("qpi_oah", np.uint8),
        ("qpi_oah_bg", np.uint8),
        ("qpi_pha", np.float32),
        ("qpi_amp", np.float32),
    ]
)
def test_bulk_qpi2d(feat, dtype):
    num = 7
    feat_data = np.zeros((20, 90, 50), dtype=dtype)
    if dtype == np.uint8:
        feat_data += np.arange(90, dtype=dtype).reshape(1, 90, 1)
    else:
        feat_data += np.linspace(0.8, 1.1, 90, dtype=dtype).reshape(1, 90, 1)
    data = {"area_um": np.linspace(100.7, 110.9, num),
            feat: feat_data}
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_bulk_qpi_")
    with RTDCWriter(rtdc_file) as hw:
        for feat in data:
            hw.store_feature(feat, data[feat])
    # Read the file:
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        events = rtdc_data["events"]
        assert feat in events.keys()
        assert np.allclose(events[feat][6], feat_data[6])
        assert events[feat][:].dtype == dtype


def test_bulk_mask():
    num = 7
    mask = []
    for ii in range(5, num + 5):
        mii = np.zeros(200, dtype=bool)
        mii[:ii] = True
        mask.append(mii.reshape(20, 10))
    data = {"area_um": np.linspace(100.7, 110.9, num),
            "mask": mask}
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_bulk_mask_")
    with RTDCWriter(rtdc_file) as hw:
        for feat in data:
            hw.store_feature(feat, data[feat])
    # Read the file:
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        events = rtdc_data["events"]
        assert "mask" in events.keys()
        # Masks are stored as uint8
        assert np.allclose(events["mask"][6], mask[6]*255)
        assert events["mask"][1].shape == (20, 10)


def test_bulk_logs():
    log = ["This is a test log that contains two lines.",
           "This is the second line.",
           ]
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_bulk_logs_")
    hw = RTDCWriter(rtdc_file)
    hw.store_log("testlog", log)
    # Read the file:
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        outlog = rtdc_data["logs"]["testlog"]
        for ii in range(len(outlog)):
            if isinstance(outlog[ii], bytes):
                # h5py 3 reads strings as bytes by default
                outii = outlog[ii].decode("utf-8")
            else:
                outii = outlog[ii]
            assert outii == log[ii]


def test_bulk_trace():
    num = 20
    trace = {"fl1_median": np.arange(num * 111).reshape(num, 111),
             "fl1_raw": 13 + np.arange(num * 111).reshape(num, 111),
             }
    data = {"area_um": np.linspace(100.7, 110.9, num),
            "trace": trace}
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_bulk_trace_")
    with RTDCWriter(rtdc_file) as hw:
        for feat in data:
            hw.store_feature(feat, data[feat])
    # Read the file:
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        events = rtdc_data["events"]
        assert "trace" in events.keys()
        assert np.allclose(events["trace"]["fl1_raw"], trace["fl1_raw"])


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_contour_from_hdf5():
    ds1 = new_dataset(retrieve_data("fmt-hdf5_image-bg_2020.zip"))
    assert ds1["contour"].shape == (5, np.nan, 2)

    edest = tempfile.mkdtemp()
    f1 = join(edest, "dclab_test_export_hdf5_image.rtdc")
    with RTDCWriter(f1) as hw:
        hw.store_metadata({"setup": ds1.config["setup"],
                           "experiment": ds1.config["experiment"]})
        hw.store_feature("deform", ds1["deform"])
        hw.store_feature("contour", ds1["contour"])

    ds2 = new_dataset(f1)
    assert ds2["contour"].shape == (5, np.nan, 2)


def test_empty_array():
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_error_")
    with RTDCWriter(rtdc_file) as hw:
        with pytest.raises(ValueError, match="Empty data object for 'deform'"):
            hw.store_feature("deform", [])


def test_index_increment():
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_error_")
    with RTDCWriter(rtdc_file) as hw:
        hw.store_metadata({"experiment": {"sample": "test",
                                          "run index": 1}})
        hw.store_feature("index", np.arange(1, 11))
        hw.store_feature("index", np.arange(1, 11))

    with new_dataset(rtdc_file) as ds:
        assert np.all(ds["index"] == np.arange(1, 21))


def test_index_increment2():
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_error_")
    with RTDCWriter(rtdc_file) as hw:
        hw.store_metadata({"experiment": {"sample": "test",
                                          "run index": 1}})
        # actually, RTDCWriter only uses the shape!
        hw.store_feature("index", np.arange(10, 20))
        hw.store_feature("index", np.arange(50, 60))

    with new_dataset(rtdc_file) as ds:
        assert np.all(ds["index"] == np.arange(1, 21))


def test_index_online_no_increment():
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_error_")
    with RTDCWriter(rtdc_file) as hw:
        hw.store_metadata({"experiment": {"sample": "test",
                                          "run index": 1}})
        hw.store_feature("index_online", np.arange(10))
        hw.store_feature("index_online", np.arange(10))

    with new_dataset(rtdc_file) as ds:
        assert np.all(ds["index_online"] == np.concatenate([np.arange(10),
                                                            np.arange(10)]))


def test_data_error():
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_error_")
    with pytest.raises(ValueError, match="unknown"):
        RTDCWriter(rtdc_file, mode="unknown")

    hw = RTDCWriter(rtdc_file)
    with pytest.raises(ValueError, match="area_undefined"):
        hw.store_feature("area_undefined", np.linspace(100.7, 110.9, 100))

    with pytest.raises(ValueError, match="fl_unknown"):
        hw.store_feature("trace", {"fl_unknown": np.arange(10)})


def test_logs_append():
    log1 = ["This is a test log that contains two lines.",
            "This is the second line.",
            ]
    log2 = ["These are other logging events.",
            "They are appended to the log.",
            "And may have different lengths."
            ]
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_append_logs_")
    hw = RTDCWriter(rtdc_file, mode="reset")
    hw.store_log("testlog", log1)
    hw.store_log("testlog", log2)

    # Read the file:
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        outlog = rtdc_data["logs"]["testlog"]
        for ii in range(len(outlog)):
            if isinstance(outlog[ii], bytes):
                # h5py 3 reads strings as bytes by default
                outii = outlog[ii].decode("utf-8")
            else:
                outii = outlog[ii]
            assert outii == (log1 + log2)[ii]


def test_meta():
    data = {"area_um": np.linspace(100.7, 110.9, 100)}
    meta = {"setup": {
        "channel width": 20,
        "chip region": "Channel",  # should be made lower-case
    },
        "online_contour": {
        "no absdiff": "True",  # should be converted to bool
        "image blur": 3.0,
    },
    }
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_meta_")
    with RTDCWriter(rtdc_file) as hw:
        for feat in data:
            hw.store_feature(feat, data[feat])
        hw.store_metadata(meta)
    # Read the file:
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        abool = rtdc_data.attrs["online_contour:no absdiff"]
        assert abool
        assert isinstance(abool, (bool, np.bool_))
        anint = rtdc_data.attrs["online_contour:image blur"]
        assert isinstance(anint, numbers.Integral)
        assert rtdc_data.attrs["setup:channel width"] == 20
        assert rtdc_data.attrs["setup:chip region"] == "channel"


def test_meta_bytes():
    data = {"area_um": np.linspace(100.7, 110.9, 100)}
    meta = {
        "setup": {
            "channel width": 20,
            "chip region": b"channel"  # bytes should be converted to str
        },
        "experiment": {
            "date": b"2020-08-12"  # bytes should be converted to str
        }}
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_meta_")
    with RTDCWriter(rtdc_file) as hw:
        for feat in data:
            hw.store_feature(feat, data[feat])
        hw.store_metadata(meta)
    # Read the file:
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        assert rtdc_data.attrs["setup:channel width"] == 20
        assert rtdc_data.attrs["setup:chip region"] == "channel"
        assert rtdc_data.attrs["experiment:date"] == "2020-08-12"


def test_meta_error():
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_error_meta_")

    meta1 = {"rediculous_section": {"a": 4}}
    hw = RTDCWriter(rtdc_file)
    with pytest.raises(ValueError, match="rediculous_section"):
        hw.store_metadata(meta1)

    meta2 = {"setup": {"rediculous_key": 4}}
    with pytest.raises(ValueError, match="rediculous_key"):
        hw.store_metadata(meta2)


def test_meta_no_analysis():
    """The "filtering" section should not be written to the dataset"""
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_meta_no_analysis")

    meta1 = {"filtering": {"enable filters": True}}
    with RTDCWriter(rtdc_file) as hw:
        hw.store_feature("area_um", np.linspace(100.7, 110.9, 100))
        with pytest.raises(ValueError, match="filtering"):
            hw.store_metadata(meta1)


def test_mode():
    data = {"area_um": np.linspace(100.7, 110.9, 100)}
    data2 = {"deform": np.linspace(.7, .8, 100)}

    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_replace_")
    with RTDCWriter(rtdc_file, mode="reset") as hw:
        for feat in data:
            hw.store_feature(feat, data[feat])

    with RTDCWriter(rtdc_file, mode="append") as hw:
        for feat in data:
            hw.store_feature(feat, data[feat])

    # Read the file:
    with h5py.File(rtdc_file, mode="r") as rtdc_data1:
        events1 = rtdc_data1["events"]
        assert "area_um" in events1.keys()
        assert len(events1["area_um"]) == 2 * len(data["area_um"])

    with RTDCWriter(rtdc_file, mode="replace") as hw:
        for feat in data:
            hw.store_feature(feat, data[feat])

    with RTDCWriter(rtdc_file, mode="replace") as hw:
        for feat in data2:
            hw.store_feature(feat, data2[feat])

    with h5py.File(rtdc_file, mode="r") as rtdc_data2:
        events2 = rtdc_data2["events"]
        assert "area_um" in events2.keys()
        assert "deform" in events2.keys()
        assert len(events2["area_um"]) == len(data["area_um"])


def test_nan_array():
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_error_")
    with RTDCWriter(rtdc_file) as hw:
        with pytest.warns(RuntimeWarning) as record:
            hw.store_feature("deform", np.full(100, np.nan))

    assert len(record) == 3
    assert record[0].message.args[0] == "All-NaN axis encountered"
    assert record[1].message.args[0] == "All-NaN axis encountered"
    assert record[2].message.args[0] == "Mean of empty slice"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_non_scalar_bad_shape():
    h5path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    exppath = h5path.with_name("exported.rtdc")

    register_temporary_feature("peterpan", is_scalar=False)
    with RTDCWriter(exppath, mode="append") as hw:
        data = np.arange(10 * 3 * 5).reshape(10, 3, 5)
        # This should work
        hw.store_feature("peterpan", data, shape=(3, 5))
        # This should not work
        with pytest.raises(ValueError, match="Bad shape"):
            hw.store_feature("peterpan", data, shape=(3, 6))
    deregister_temporary_feature("peterpan")  # cleanup


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_open_from_h5group():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    with h5py.File(path, "a") as h5:
        with dclab.RTDCWriter(h5) as hw:
            assert not hw.owns_path
            hw.store_log("peter", "hans")
        # do it again (to make sure h5 is not closed)
        with dclab.RTDCWriter(h5) as hw:
            assert not hw.owns_path
            hw.store_log("peter", "gert")
    # sanity check
    with dclab.new_dataset(path) as ds:
        assert "".join(ds.logs["peter"]) == "hansgert"


def test_real_time():
    # Create huge array
    n = 116
    # Writing 10 images at a time is faster than writing one image at a time
    m = 4
    assert n // m == np.round(n / m)
    shx = 48
    shy = 32
    contours = [np.arange(20).reshape(10, 2)] * m
    images = np.zeros((m, shy, shx), dtype=np.uint8)
    masks = np.zeros((m, shy, shx), dtype=np.bool_)
    traces = {"fl1_median": np.arange(m * 55).reshape(m, 55)}
    axis1 = np.linspace(0, 1, m)
    axis2 = np.arange(float(m))
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_realtime_")

    with RTDCWriter(rtdc_file, mode="reset") as hw:
        # simulate real time and write one image at a time
        for ii in range(n // m):
            # print(ii)
            num_img = np.copy(images) + ii

            data = {"area_um": axis1,
                    "area_cvx": axis2,
                    "image": num_img,
                    "contour": contours,
                    "mask": masks,
                    "trace": traces}
            for feat in data:
                hw.store_feature(feat, data[feat])

    # Read the file:
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        events = rtdc_data["events"]
        assert events["image"].shape == (n, shy, shx)
        assert events["area_um"].shape == (n,)
        assert events["contour"]["0"].shape == (10, 2)
        assert events["trace"]["fl1_median"].shape == (n, 55)
        assert np.dtype(events["area_um"]) == float
        assert np.dtype(events["area_cvx"]) == float


def test_real_time_single():
    # Create huge array
    n = 33
    shx = 30
    shy = 10
    image = np.zeros((shy, shx), dtype=np.uint8)
    mask = np.zeros((shy, shx), dtype=np.bool_)
    contour = np.arange(22).reshape(11, 2)
    trace = {"fl1_median": np.arange(43)}

    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_realtime_single_")
    with RTDCWriter(rtdc_file, mode="reset") as hw:
        # simulate real time and write one event at a time
        for ii in range(n):
            data = {"area_um": ii * .1,
                    "area_cvx": ii * 5.,
                    "image": image * ii,
                    "contour": contour,
                    "mask": mask,
                    "trace": trace}
            for feat in data:
                hw.store_feature(feat, data[feat])
            hw.store_log("log1", f"line {ii}")

    # Read the file:
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        events = rtdc_data["events"]
        assert events["image"].shape == (n, shy, shx)
        assert events["area_um"].shape == (n,)
        assert events["contour"]["0"].shape == (11, 2)
        assert events["trace"]["fl1_median"].shape == (n, 43)
        assert np.dtype(events["area_um"]) == float
        assert np.dtype(events["area_cvx"]) == float
        logs = rtdc_data["logs"]
        assert len(logs["log1"]) == n


def test_rectify_metadata_ignore_empty_image():
    # test introduced in 0.39.7
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_error_")
    with h5py.File(rtdc_file, "a") as h5:
        h5.require_group("events")
        h5["events"].require_dataset(name="image",
                                     shape=(0, 80, 100),
                                     dtype=np.uint8)
        h5["events/deform"] = np.linspace(.1, .12, 7)

        # Initialize writer
        hw = RTDCWriter(h5, mode="append")
        # in previous versions, this did not work, because of the empty trace
        hw.rectify_metadata()
        # make sure that something happened
        assert h5.attrs["experiment:event count"] == 7


def test_rectify_metadata_ignore_empty_trace():
    # test introduced in 0.39.7
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_error_")
    with h5py.File(rtdc_file, "a") as h5:
        h5.require_group("events")
        h5.require_group("events/trace")
        h5["events/deform"] = np.linspace(.1, .12, 7)

        # Initialize writer
        hw = RTDCWriter(h5, mode="append")
        # in previous versions, this did not work, because of the empty trace
        hw.rectify_metadata()
        # make sure that something happened
        assert h5.attrs["experiment:event count"] == 7


def test_replace_contour():
    num = 7
    contour = []
    contour2 = []
    for ii in range(5, num + 5):
        cii = np.arange(2 * ii).reshape(2, ii)
        contour.append(cii)
        contour2.append(cii * 2)

    data1 = {"area_um": np.linspace(100.7, 110.9, num),
             "contour": contour}
    data2 = {"contour": contour2}
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_replace_contour_")

    with RTDCWriter(rtdc_file) as hw:
        for feat in data1:
            hw.store_feature(feat, data1[feat])

    with RTDCWriter(rtdc_file, mode="replace") as hw:
        for feat in data2:
            hw.store_feature(feat, data2[feat])

    # verify
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        events = rtdc_data["events"]
        assert "contour" in events.keys()
        assert not np.allclose(events["contour"]["6"], contour[6])
        assert np.allclose(events["contour"]["6"], contour2[6])


def test_replace_logs():
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_replace_logs_")

    with RTDCWriter(rtdc_file) as hw:
        hw.store_log("log1", ["hans", "und"])

    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        logs = rtdc_data["logs"]
        assert len(logs["log1"]) == 2

    with RTDCWriter(rtdc_file, mode="replace") as hw:
        hw.store_log("log1", ["peter"])

    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        logs = rtdc_data["logs"]
        assert len(logs["log1"]) == 1


def test_scalar_ufuncs_attrs():
    """Since version 0.46.7 we support caching min/max/mean as h5 attributes"""
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_replace_logs_")
    deform = np.linspace(.1, .2, 254)
    area_um = np.linspace(200, 500, 254)
    with RTDCWriter(rtdc_file) as hw:
        hw.store_feature("deform", deform)
        hw.store_feature("area_um", area_um)

    with h5py.File(rtdc_file) as h5:
        assert np.allclose(h5["events/deform"].attrs["min"], 0.1)
        assert np.allclose(h5["events/deform"].attrs["max"], 0.2)
        assert np.allclose(h5["events/deform"].attrs["mean"], 0.15)

        assert np.allclose(h5["events/area_um"].attrs["min"], 200)
        assert np.allclose(h5["events/area_um"].attrs["max"], 500)
        assert np.allclose(h5["events/area_um"].attrs["mean"], 350)


def test_scalar_ufuncs_attrs_append():
    """Since version 0.46.7 we support caching min/max/mean as h5 attributes"""
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_replace_logs_")
    deform_a = np.linspace(.1, .2, 254)
    area_um_a = np.linspace(200, 500, 254)
    deform_b = np.linspace(.05, .1, 123)
    area_um_b = np.linspace(200, 700, 123)

    with RTDCWriter(rtdc_file) as hw:
        hw.store_feature("deform", deform_a)
        hw.store_feature("area_um", area_um_a)

    with RTDCWriter(rtdc_file) as hw:
        hw.store_feature("deform", deform_b)
        hw.store_feature("area_um", area_um_b)

    with h5py.File(rtdc_file) as h5:
        assert np.allclose(h5["events/deform"].attrs["min"], 0.05)
        assert np.allclose(h5["events/deform"].attrs["max"], 0.2)
        assert np.allclose(h5["events/deform"].attrs["mean"],
                           0.12553050397877985)

        assert np.allclose(h5["events/area_um"].attrs["min"], 200)
        assert np.allclose(h5["events/area_um"].attrs["max"], 700)
        assert np.allclose(h5["events/area_um"].attrs["mean"],
                           382.6259946949602)


def test_tables_recarray():
    """Write a numpy recarray table to an .rtdc file"""
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_tables_recarray")

    # generate a table
    columns = ["bread", "beer", "chocolate"]
    ds_dt = np.dtype({'names': columns,
                      'formats': [np.float64] * len(columns)})
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

    # Write the table to the .rtdc file
    with RTDCWriter(rtdc_file) as hw:
        hw.store_table("food", rec_arr)

    # Check whether this worked
    with h5py.File(rtdc_file) as h5:
        assert np.all(h5["tables"]["food"]["bread"].flat == np.arange(10))


def test_tables_recarray_from_h5array():
    """Write a numpy recarray table to an .rtdc file read from another file"""
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_tables_recarray")
    rtdc_file2 = tempfile.mktemp(suffix=".rtdc",
                                 prefix="dclab_test_tables_recarray")

    # generate a table
    columns = ["bread", "beer", "chocolate"]
    ds_dt = np.dtype({'names': columns,
                      'formats': [np.float64] * len(columns)})
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

    # Write the table to the .rtdc file
    with RTDCWriter(rtdc_file) as hw:
        hw.store_table("food", rec_arr)

    # Write a copy of that table directly to another HDF5 file
    with h5py.File(rtdc_file) as h5, RTDCWriter(rtdc_file2) as hw2:
        hw2.store_table("food2", h5["tables"]["food"])

    # Check whether that worked
    with h5py.File(rtdc_file2) as h52:
        assert np.all(h52["tables"]["food2"]["bread"].flat == np.arange(10))


def test_tables_recarray_from_dict():
    """Write a numpy recarray table to an .rtdc file"""
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_tables_recarray")

    # generate a table
    tab_data = {
        "bread": np.arange(10),
        "beer": np.full(10, 1000),
        "chocolate": np.linspace(np.pi, 2*np.pi, 10)
    }

    # Write the table to the .rtdc file
    with RTDCWriter(rtdc_file) as hw:
        hw.store_table("food", tab_data)

    # Check whether this worked
    with h5py.File(rtdc_file) as h5:
        assert np.all(h5["tables"]["food"]["bread"].flat == np.arange(10))


def test_tables_recarray_from_dict_with_lists():
    """Write a numpy recarray table to an .rtdc file"""
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_tables_recarray")

    # generate a table
    tab_data = {
        "bread": np.arange(10).tolist(),
        "beer": np.full(10, 1000).tolist(),
        "chocolate": np.linspace(np.pi, 2*np.pi, 10).tolist()
    }

    # Write the table to the .rtdc file
    with RTDCWriter(rtdc_file) as hw:
        hw.store_table("food", tab_data)

    # Check whether this worked
    with h5py.File(rtdc_file) as h5:
        assert np.all(h5["tables"]["food"]["bread"].flat == np.arange(10))


def test_tables_recarray_invalid():
    """Write a numpy recarray table to an .rtdc file"""
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_tables_recarray")

    # generate an invalid table
    tab_data = [np.arange(10),
                np.full(10, 1000),
                np.linspace(np.pi, 2*np.pi, 10),
                ]

    # Write the table to the .rtdc file
    with RTDCWriter(rtdc_file) as hw:
        with pytest.raises(NotImplementedError, match="list"):
            hw.store_table("food", tab_data)


def test_version_branding_1_write_single_version():
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_branding_")

    this_version = f"dclab {dclab.__version__}"

    with RTDCWriter(rtdc_file) as hw:
        hw.store_feature("deform", np.linspace(0.01, 0.02, 10))

    with h5py.File(rtdc_file) as h5:
        assert h5.attrs["setup:software version"] == this_version


def test_version_branding_2_dont_override_initial_version():
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_branding_")

    this_version = f"dclab {dclab.__version__}"

    with h5py.File(rtdc_file, "a") as h5:
        h5.attrs["setup:software version"] = "Shape-In 1.2.3"

    with RTDCWriter(rtdc_file) as hw:
        hw.store_feature("deform", np.linspace(0.01, 0.02, 10))

    expected_version = f"Shape-In 1.2.3 | {this_version}"

    with h5py.File(rtdc_file) as h5:
        assert h5.attrs["setup:software version"] == expected_version


def test_version_branding_3_use_old_version_manual():
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_branding_")

    this_version = f"dclab {dclab.__version__}"

    with h5py.File(rtdc_file, "a") as h5:
        h5.attrs["setup:software version"] = "Shape-In 1.2.3"

    with RTDCWriter(rtdc_file) as hw:
        hw.store_feature("deform", np.linspace(0.01, 0.02, 10))
        hw.version_brand(old_version="Peter 1.0")

    expected_version = f"Peter 1.0 | {this_version}"

    with h5py.File(rtdc_file) as h5:
        assert h5.attrs["setup:software version"] == expected_version


def test_version_branding_4_dont_write_attribute():
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_branding_")

    this_version = f"dclab {dclab.__version__}"

    with h5py.File(rtdc_file, "a") as h5:
        h5.attrs["setup:software version"] = "Shape-In 1.2.3"

    with RTDCWriter(rtdc_file) as hw:
        hw.store_feature("deform", np.linspace(0.01, 0.02, 10))
        hw.version_brand(old_version="Peter 1.0", write_attribute=False)

    expected_version = f"Shape-In 1.2.3 | {this_version}"

    with h5py.File(rtdc_file) as h5:
        assert h5.attrs["setup:software version"] == expected_version
