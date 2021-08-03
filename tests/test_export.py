import io
import os
from os.path import join
import tempfile

import h5py
import numpy as np
import pytest

import dclab
from dclab import dfn, new_dataset

from helper_methods import example_data_dict, retrieve_data


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.exc.CorruptFrameWarning')
def test_avi_export():
    pytest.importorskip("nptdms")
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))
    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.avi")
    ds.export.avi(path=f1)
    assert os.stat(
        f1)[6] > 1e4, "Resulting file to small, Something went wrong!"


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.exc.CorruptFrameWarning')
def test_avi_override():
    pytest.importorskip("nptdms")
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.avi")
    ds.export.avi(f1, override=True)
    try:
        ds.export.avi(f1[:-4], override=False)
    except OSError:
        pass
    else:
        raise ValueError("Should append .avi and not override!")


def test_avi_no_images():
    pytest.importorskip("imageio")
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=127, keys=keys)
    ds = dclab.new_dataset(ddict)

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.avi")
    try:
        ds.export.avi(f1)
    except OSError:
        pass
    else:
        raise ValueError("There should be no image data to write!")


def test_fcs_export():
    pytest.importorskip("fcswrite")
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=222, keys=keys)
    ds = dclab.new_dataset(ddict)

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.fcs")
    f2 = join(edest, "test_unicode.fcs")

    ds.export.fcs(f1, keys, override=True)
    ds.export.fcs(f2, [u"area_um", u"deform", u"time",
                       u"frame", u"fl3_width"], override=True)

    with io.open(f1, mode="rb") as fd:
        a1 = fd.read()

    with io.open(f2, mode="rb") as fd:
        a2 = fd.read()

    assert a1 == a2
    assert len(a1) != 0


def test_fcs_override():
    pytest.importorskip("fcswrite")
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=212, keys=keys)
    ds = dclab.new_dataset(ddict)

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.fcs")
    ds.export.fcs(f1, keys, override=True)
    try:
        ds.export.fcs(f1[:-4], keys, override=False)
    except OSError:
        pass
    else:
        raise ValueError("Should append .fcs and not override!")


def test_fcs_not_filtered():
    pytest.importorskip("fcswrite")
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=127, keys=keys)
    ds = dclab.new_dataset(ddict)

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.tsv")
    ds.export.fcs(f1, keys, filtered=False)


def test_hdf5():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=127, keys=keys)
    ds1 = dclab.new_dataset(ddict)
    ds1.config["experiment"]["sample"] = "test"
    ds1.config["experiment"]["run index"] = 1
    ds1.config["imaging"]["frame rate"] = 2000

    edest = tempfile.mkdtemp()
    f1 = join(edest, "dclab_test_export_hdf5.rtdc")
    ds1.export.hdf5(f1, keys)

    ds2 = dclab.new_dataset(f1)
    assert ds1 != ds2
    assert np.allclose(ds2["area_um"], ds1["area_um"])
    assert np.allclose(ds2["deform"], ds1["deform"])
    assert np.allclose(ds2["time"], ds1["time"])
    assert np.allclose(ds2["frame"], ds1["frame"])
    assert np.allclose(ds2["fl3_width"], ds1["fl3_width"])


def test_hdf5_contour_image_trace():
    n = 65
    keys = ["contour", "image", "trace"]
    ddict = example_data_dict(size=n, keys=keys)

    ds1 = dclab.new_dataset(ddict)
    ds1.config["experiment"]["sample"] = "test"
    ds1.config["experiment"]["run index"] = 1

    edest = tempfile.mkdtemp()
    f1 = join(edest, "dclab_test_export_hdf5_image.rtdc")
    ds1.export.hdf5(f1, keys, filtered=False)
    ds2 = dclab.new_dataset(f1)

    for ii in range(n):
        assert np.all(ds1["image"][ii] == ds2["image"][ii])
        assert np.all(ds1["contour"][ii] == ds2["contour"][ii])

    for key in dfn.FLUOR_TRACES:
        assert np.all(ds1["trace"][key] == ds2["trace"][key])


def test_hdf5_contour_image_trace_large():
    """Same test for large event numbers (to test chunking)"""
    n = 653
    keys = ["contour", "image", "trace"]
    ddict = example_data_dict(size=n, keys=keys)

    ds1 = dclab.new_dataset(ddict)
    ds1.config["experiment"]["sample"] = "test"
    ds1.config["experiment"]["run index"] = 1

    edest = tempfile.mkdtemp()
    f1 = join(edest, "dclab_test_export_hdf5_image.rtdc")
    ds1.export.hdf5(f1, keys, filtered=False)
    ds2 = dclab.new_dataset(f1)

    # the trace may have negative values, it's int16, not uint16.
    assert ds1["trace"]["fl1_median"][0][0] == -1, "sanity check"

    for ii in range(n):
        assert np.all(ds1["image"][ii] == ds2["image"][ii])
        assert np.all(ds1["contour"][ii] == ds2["contour"][ii])

    for key in dfn.FLUOR_TRACES:
        assert np.all(ds1["trace"][key] == ds2["trace"][key])


def test_hdf5_filtered():
    n = 10
    keys = ["area_um", "image"]
    ddict = example_data_dict(size=n, keys=keys)
    ddict["image"][3] = np.arange(10 * 20, dtype=np.uint8).reshape(10, 20) + 22

    ds1 = dclab.new_dataset(ddict)
    ds1.config["experiment"]["sample"] = "test"
    ds1.config["experiment"]["run index"] = 1
    ds1.filter.manual[2] = False
    ds1.apply_filter()
    fta = ds1.filter.manual.copy()

    edest = tempfile.mkdtemp()
    f1 = join(edest, "dclab_test_export_hdf5_filtered.rtdc")
    ds1.export.hdf5(f1, keys)

    ds2 = dclab.new_dataset(f1)

    assert ds1 != ds2
    assert np.allclose(ds2["area_um"], ds1["area_um"][fta])
    assert np.allclose(ds2["image"][2], ds1["image"][3])
    assert np.all(ds2["image"][2] != ds1["image"][2])


def test_hdf5_filtered_index():
    """Make sure that exported index is always re-enumerated"""
    n = 10
    keys = ["area_um", "deform", "index"]
    ddict = example_data_dict(size=n, keys=keys)
    ddict["index"] = np.arange(1, n+1)

    ds1 = dclab.new_dataset(ddict)
    ds1.config["experiment"]["sample"] = "test"
    ds1.config["experiment"]["run index"] = 1
    ds1.filter.manual[2] = False
    ds1.apply_filter()

    edest = tempfile.mkdtemp()
    f1 = join(edest, "dclab_test_export_hdf5_filtered.rtdc")
    ds1.export.hdf5(f1, keys)

    ds2 = dclab.new_dataset(f1)

    assert len(ds2) == n - 1
    assert np.all(ds2["index"] == np.arange(1, n))
    assert ds2.config["experiment"]["event count"] == n - 1


def test_hdf5_frame():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=10, keys=keys)
    ds1 = dclab.new_dataset(ddict)
    ds1.config["experiment"]["sample"] = "test"
    ds1.config["experiment"]["run index"] = 1
    ds1.config["imaging"]["frame rate"] = 2000

    edest = tempfile.mkdtemp()
    f1 = join(edest, "dclab_test_export_hdf5.rtdc")
    ds1.export.hdf5(f1, keys)
    with h5py.File(f1, "a") as h5:
        for feat in keys:
            dclab.rtdc_dataset.export.hdf5_append(h5obj=h5,
                                                  rtdc_ds=ds1,
                                                  feat=feat,
                                                  time_offset=10,
                                                  compression="gzip")
    # make sure that "frame" in f1 is continuous
    ds2 = dclab.new_dataset(f1)
    assert ds2["frame"][10] == 20000


def test_hdf5_image_bg():
    n = 65
    keys = ["image", "image_bg"]
    ddict = example_data_dict(size=n, keys=keys)

    ds1 = dclab.new_dataset(ddict)
    ds1.config["experiment"]["sample"] = "test"
    ds1.config["experiment"]["run index"] = 1

    edest = tempfile.mkdtemp()
    f1 = join(edest, "dclab_test_export_hdf5_image_bg.rtdc")
    ds1.export.hdf5(f1, keys, filtered=False)
    ds2 = dclab.new_dataset(f1)

    for ii in range(n):
        assert np.all(ds1["image"][ii] == ds2["image"][ii])
        assert np.all(ds1["image_bg"][ii] == ds2["image_bg"][ii])


def test_hdf5_index_continuous():
    keys = ["area_um", "deform", "time", "frame", "index"]
    ddict = example_data_dict(size=10, keys=keys)
    ds1 = dclab.new_dataset(ddict)
    ds1.config["experiment"]["sample"] = "test"
    ds1.config["experiment"]["run index"] = 1
    ds1.config["imaging"]["frame rate"] = 2000

    edest = tempfile.mkdtemp()
    f1 = join(edest, "dclab_test_export_hdf5.rtdc")
    ds1.export.hdf5(f1, keys)
    with h5py.File(f1, "a") as h5:
        for feat in keys:
            dclab.rtdc_dataset.export.hdf5_append(h5obj=h5,
                                                  rtdc_ds=ds1,
                                                  feat=feat,
                                                  time_offset=10,
                                                  compression="gzip")
    with h5py.File(f1, "r") as h5:
        assert "index" in h5["events"]
        assert np.allclose(h5["events/index"][:], np.arange(1, 21))


def test_hdf5_index_online_continuous():
    keys = ["area_um", "deform", "time", "frame", "index_online"]
    ddict = example_data_dict(size=10, keys=keys)
    ds1 = dclab.new_dataset(ddict)
    ds1.config["experiment"]["sample"] = "test"
    ds1.config["experiment"]["run index"] = 1
    ds1.config["imaging"]["frame rate"] = 2000

    edest = tempfile.mkdtemp()
    f1 = join(edest, "dclab_test_export_hdf5.rtdc")
    ds1.export.hdf5(f1, keys)
    with h5py.File(f1, "a") as h5:
        for feat in keys:
            dclab.rtdc_dataset.export.hdf5_append(h5obj=h5,
                                                  rtdc_ds=ds1,
                                                  feat=feat,
                                                  time_offset=10,
                                                  compression="gzip")
    with h5py.File(f1, "r") as h5:
        assert "index_online" in h5["events"]


def test_hdf5_index_online_replaces_index():
    keys = ["area_um", "deform", "time", "frame", "index_online"]
    ddict = example_data_dict(size=10, keys=keys)
    ds1 = dclab.new_dataset(ddict)
    ds1.config["experiment"]["sample"] = "test"
    ds1.config["experiment"]["run index"] = 1
    ds1.config["imaging"]["frame rate"] = 2000

    edest = tempfile.mkdtemp()
    f1 = join(edest, "dclab_test_export_hdf5.rtdc")
    ds1.export.hdf5(f1, keys)
    with h5py.File(f1, "r") as h5:
        assert "index" not in h5["events"]
        assert "index_online" in h5["events"]


def test_hdf5_ml_score():
    data = {"ml_score_ds9": [.80, .31, .12, .01, .59, .40, .52],
            "ml_score_dsc": [.12, .52, .21, .24, .30, .22, .79],
            "ml_score_ent": [.42, .11, .78, .11, .54, .24, .15],
            "ml_score_pic": [.30, .30, .10, .99, .59, .55, .19],
            "ml_score_tng": [.13, .33, .13, .01, .79, .11, .22],
            "ml_score_tos": [.14, .34, .12, .01, .59, .56, .17],
            "ml_score_voy": [.25, .12, .42, .33, .21, .55, .82],
            }
    ds0 = dclab.new_dataset(data)
    ds0.config["experiment"]["sample"] = "test"
    ds0.config["experiment"]["run index"] = 1

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.rtdc")
    ds0.export.hdf5(f1, list(data.keys()), override=True)

    with dclab.new_dataset(f1) as ds:
        assert np.allclose(ds["ml_class"], np.arange(7))
        assert "ml_class" not in ds._events
        assert "ml_class" in ds._ancillaries


def test_hdf5_override():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=212, keys=keys)
    ds = dclab.new_dataset(ddict)
    ds.config["imaging"]["frame rate"] = 2000

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.rtdc")
    ds.export.hdf5(f1, keys, override=True)
    try:
        ds.export.hdf5(f1[:-5], keys, override=False)
    except OSError:
        pass
    else:
        raise ValueError("Should append .rtdc and not override!")


def test_hdf5_trace_from_tdms():
    pytest.importorskip("nptdms")
    ds = new_dataset(retrieve_data("fmt-tdms_2fl-no-image_2017.zip"))

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.rtdc")
    ds.export.hdf5(f1, ["trace"])

    ds2 = new_dataset(f1)

    for key in ds["trace"]:
        for ii in range(len(ds)):
            assert np.all(ds["trace"][key][ii] == ds2["trace"][key][ii])


def test_hdf5_traces():
    """Length of traces is preserved (no filtering)"""
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))

    # sanity check
    assert len(ds) == 7
    assert len(ds["trace"]["fl1_median"]) == 7

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.rtdc")
    ds.export.hdf5(f1, ["deform", "trace"])

    ds2 = new_dataset(f1)
    assert len(ds2) == 7
    assert len(ds2["trace"]["fl1_median"]) == 7


def test_hdf5_traces_filter():
    """Length of traces was wrong when filters were applied #112

    Test dataset length with additional feature.
    """
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))

    # applying some filters
    ds.config["filtering"]["deform min"] = 0.01
    ds.config["filtering"]["deform max"] = 0.1
    ds.apply_filter()

    # sanity check
    assert np.sum(ds.filter.all) == 3

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.rtdc")
    ds.export.hdf5(f1, ["deform", "trace"])

    ds2 = new_dataset(f1)
    assert len(ds2) == 3
    assert len(ds2["deform"]) == 3
    assert len(ds2["trace"]["fl1_median"]) == 3
    assert np.all(ds["trace"]["fl1_raw"][3] == ds2["trace"]["fl1_raw"][0])
    assert np.all(ds["trace"]["fl1_raw"][5] == ds2["trace"]["fl1_raw"][1])
    assert np.all(ds["trace"]["fl1_raw"][6] == ds2["trace"]["fl1_raw"][2])


def test_hdf5_traces_filter2():
    """Length of traces was wrong when filters were applied #112

    Test dataset lenght only with trace.
    """
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))

    # applying some filters
    ds.config["filtering"]["deform min"] = 0.01
    ds.config["filtering"]["deform max"] = 0.1
    ds.apply_filter()

    # sanity check
    assert np.sum(ds.filter.all) == 3

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.rtdc")
    ds.export.hdf5(f1, ["deform", "trace"])

    ds2 = new_dataset(f1)
    assert len(ds2) == 3
    assert len(ds2["deform"]) == 3
    assert len(ds2["trace"]["fl1_median"]) == 3
    assert np.all(ds["trace"]["fl1_raw"][3] == ds2["trace"]["fl1_raw"][0])
    assert np.all(ds["trace"]["fl1_raw"][5] == ds2["trace"]["fl1_raw"][1])
    assert np.all(ds["trace"]["fl1_raw"][6] == ds2["trace"]["fl1_raw"][2])


def test_tsv_export():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=222, keys=keys)
    ds = dclab.new_dataset(ddict)

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.tsv")
    f2 = join(edest, "test_unicode.tsv")

    ds.export.tsv(f1, keys, override=True)
    ds.export.tsv(f2, [u"area_um", u"deform", u"time",
                       u"frame", u"fl3_width"], override=True)

    with io.open(f1) as fd:
        a1 = fd.read()

    with io.open(f2) as fd:
        a2 = fd.read()

    assert a1 == a2
    assert len(a1) != 0


def test_tsv_override():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=212, keys=keys)
    ds = dclab.new_dataset(ddict)

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.tsv")
    ds.export.tsv(f1, keys, override=True)
    try:
        ds.export.tsv(f1[:-4], keys, override=False)
    except OSError:
        pass
    else:
        raise ValueError("Should append .tsv and not override!")


def test_tsv_not_filtered():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=127, keys=keys)
    ds = dclab.new_dataset(ddict)

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.tsv")
    ds.export.tsv(f1, keys, filtered=False)


if __name__ == "__main__":
    # Run all tests
    _loc = locals()
    for _key in list(_loc.keys()):
        if _key.startswith("test_") and hasattr(_loc[_key], "__call__"):
            _loc[_key]()
