import numbers
import tempfile

import h5py
import numpy as np

from dclab.rtdc_dataset import write


def test_bulk_scalar():
    data = {"area_um": np.linspace(100.7, 110.9, 100)}
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_bulk_scalar_")
    write(rtdc_file, data)
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
    write(rtdc_file, data)
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
    write(rtdc_file, data)
    # Read the file:
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        events = rtdc_data["events"]
        assert "image" in events.keys()
        assert np.allclose(events["image"][6], image[6])


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
    write(rtdc_file, data)
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
    write(rtdc_file, logs={"testlog": log})
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
    write(rtdc_file, data)
    # Read the file:
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        events = rtdc_data["events"]
        assert "trace" in events.keys()
        assert np.allclose(events["trace"]["fl1_raw"], trace["fl1_raw"])


def test_data_error():
    data = {"area_um": np.linspace(100.7, 110.9, 100)}
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_error_")
    try:
        write(rtdc_file, data, mode="unknown")
    except ValueError:
        pass
    else:
        assert False, "ValueError should have been raised (wrong mode)"

    try:
        write(rtdc_file, ["peter"])
    except ValueError:
        pass
    else:
        assert False, "ValueError should have been raised (wrong data)"

    data2 = {"area_undefined": np.linspace(100.7, 110.9, 100)}
    try:
        write(rtdc_file, data2)
    except ValueError:
        pass
    else:
        assert False, "ValueError should have been raised (feature name)"

    data3 = {"trace": {"fl_unknown": np.arange(10)}}
    try:
        write(rtdc_file, data3)
    except ValueError:
        pass
    else:
        assert False, "ValueError should have been raised (trace name)"


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
    with h5py.File(rtdc_file, "w") as fobj:
        write(fobj, logs={"testlog": log1}, mode="append")
        write(fobj, logs={"testlog": log2}, mode="append")
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
    write(rtdc_file, data, meta=meta)
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
    write(rtdc_file, data, meta=meta)
    # Read the file:
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        assert rtdc_data.attrs["setup:channel width"] == 20
        assert rtdc_data.attrs["setup:chip region"] == "channel"
        assert rtdc_data.attrs["experiment:date"] == "2020-08-12"


def test_meta_error():
    data = {"area_um": np.linspace(100.7, 110.9, 100)}
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_error_meta_")

    meta1 = {"rediculous_section": {"a": 4}}
    try:
        write(rtdc_file, data, meta=meta1)
    except ValueError:
        pass
    else:
        assert False, "ValueError should have been raised (unknown section)"

    meta2 = {"setup": {"rediculous_key": 4}}
    try:
        write(rtdc_file, data, meta=meta2)
    except ValueError:
        pass
    else:
        assert False, "ValueError should have been raised (unknown key)"


def test_meta_no_analysis():
    """The "filtering" section should not be written to the dataset"""
    data = {"area_um": np.linspace(100.7, 110.9, 100)}
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_meta_no_analysis")

    meta1 = {"filtering": {"enable filters": True}}
    try:
        write(rtdc_file, data, meta=meta1)
    except ValueError:
        pass
    else:
        assert False, "ValueError should have been raised (unknown section)"


def test_mode():
    data = {"area_um": np.linspace(100.7, 110.9, 100)}
    data2 = {"deform": np.linspace(.7, .8, 100)}

    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_replace_")
    write(rtdc_file, data=data, mode="reset")
    write(rtdc_file, data=data, mode="append").close()
    # Read the file:
    with h5py.File(rtdc_file, mode="r") as rtdc_data1:
        events1 = rtdc_data1["events"]
        assert "area_um" in events1.keys()
        assert len(events1["area_um"]) == 2 * len(data["area_um"])

    write(rtdc_file, data=data, mode="replace")
    write(rtdc_file, data=data2, mode="replace")
    with h5py.File(rtdc_file, mode="r") as rtdc_data2:
        events2 = rtdc_data2["events"]
        assert "area_um" in events2.keys()
        assert "deform" in events2.keys()
        assert len(events2["area_um"]) == len(data["area_um"])


def test_mode_return():
    data = {"area_um": np.linspace(100.7, 110.9, 100)}

    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_return_")

    ret1 = write(rtdc_file, data=data, mode="append")
    assert isinstance(ret1, h5py.File)
    ret1.close()

    ret2 = write(rtdc_file, data=data, mode="replace")
    assert ret2 is None

    ret3 = write(rtdc_file, data=data, mode="reset")
    assert ret3 is None


def test_real_time():
    # Create huge array
    N = 116
    # Writing 10 images at a time is faster than writing one image at a time
    M = 4
    assert N // M == np.round(N / M)
    shx = 48
    shy = 32
    contours = [np.arange(20).reshape(10, 2)] * M
    images = np.zeros((M, shy, shx), dtype=np.uint8)
    masks = np.zeros((M, shy, shx), dtype=np.bool_)
    traces = {"fl1_median": np.arange(M * 55).reshape(M, 55)}
    axis1 = np.linspace(0, 1, M)
    axis2 = np.arange(float(M))
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_realtime_")
    with h5py.File(rtdc_file, "w") as fobj:
        # simulate real time and write one image at a time
        for ii in range(N // M):
            # print(ii)
            num_img = np.copy(images) + ii

            data = {"area_um": axis1,
                    "area_cvx": axis2,
                    "image": num_img,
                    "contour": contours,
                    "mask": masks,
                    "trace": traces}

            write(fobj,
                  data=data,
                  mode="append",
                  )

    # Read the file:
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        events = rtdc_data["events"]
        assert events["image"].shape == (N, shy, shx)
        assert events["area_um"].shape == (N,)
        assert events["contour"]["0"].shape == (10, 2)
        assert events["trace"]["fl1_median"].shape == (N, 55)
        assert np.dtype(events["area_um"]) == float
        assert np.dtype(events["area_cvx"]) == float


def test_real_time_single():
    # Create huge array
    N = 33
    shx = 30
    shy = 10
    image = np.zeros((shy, shx), dtype=np.uint8)
    mask = np.zeros((shy, shx), dtype=np.bool_)
    contour = np.arange(22).reshape(11, 2)
    trace = {"fl1_median": np.arange(43)}

    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_realtime_single_")
    with h5py.File(rtdc_file, "w") as fobj:
        # simulate real time and write one image at a time
        for ii in range(N):
            data = {"area_um": ii * .1,
                    "area_cvx": ii * 5.,
                    "image": image * ii,
                    "contour": contour,
                    "mask": mask,
                    "trace": trace}
            write(fobj,
                  data=data,
                  mode="append",
                  logs={"log1": "line {}".format(ii)}
                  )

    # Read the file:
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        events = rtdc_data["events"]
        assert events["image"].shape == (N, shy, shx)
        assert events["area_um"].shape == (N,)
        assert events["contour"]["0"].shape == (11, 2)
        assert events["trace"]["fl1_median"].shape == (N, 43)
        assert np.dtype(events["area_um"]) == float
        assert np.dtype(events["area_cvx"]) == float
        logs = rtdc_data["logs"]
        assert len(logs["log1"]) == N


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
    write(rtdc_file, data1)
    write(rtdc_file, data2, mode="replace")
    # verify
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        events = rtdc_data["events"]
        assert "contour" in events.keys()
        assert not np.allclose(events["contour"]["6"], contour[6])
        assert np.allclose(events["contour"]["6"], contour2[6])


def test_replace_logs():
    rtdc_file = tempfile.mktemp(suffix=".rtdc",
                                prefix="dclab_test_replace_logs_")
    write(rtdc_file, logs={"log1": ["hans", "und"]})
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        logs = rtdc_data["logs"]
        assert len(logs["log1"]) == 2

    write(rtdc_file, logs={"log1": ["peter"]}, mode="replace")
    with h5py.File(rtdc_file, mode="r") as rtdc_data:
        logs = rtdc_data["logs"]
        assert len(logs["log1"]) == 1


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
