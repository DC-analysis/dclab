"""Test filter hierarchies"""

import h5py
import numpy as np
import pytest

from dclab import new_dataset
from dclab.rtdc_dataset import fmt_hierarchy, write

from helper_methods import example_data_dict, retrieve_data


@pytest.mark.filterwarnings('ignore::dclab.features.emodulus.'
                            + 'YoungsModulusLookupTableExceededWarning')
def test_config_calculation():
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ds.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                "emodulus medium": "CellCarrier",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    ch = new_dataset(ds)
    assert np.allclose(ch["emodulus"], ds["emodulus"], equal_nan=True)
    ds.config["calculation"]["emodulus temperature"] = 24.0
    assert np.allclose(ch["emodulus"], ds["emodulus"], equal_nan=True)
    # the user still has to call `apply_filter` to update the config:
    assert not ch.config["calculation"]["emodulus temperature"] == 24.0
    ch.apply_filter()
    assert ch.config["calculation"]["emodulus temperature"] == 24.0


def test_event_count():
    pytest.importorskip("nptdms")
    tdms_path = retrieve_data("fmt-tdms_fl-image_2016.zip")
    ds = new_dataset(tdms_path)
    ds.filter.manual[0] = False
    ch = new_dataset(ds)
    assert ds.config["experiment"]["event count"] == len(ds)
    assert ch.config["experiment"]["event count"] == len(ch)
    assert len(ds) == len(ch) + 1


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_feat_contour():
    path = retrieve_data("fmt-hdf5_fl_2017.zip")
    ds = new_dataset(path)
    ds.filter.manual[0] = False
    ds.filter.manual[2] = False
    ch = new_dataset(ds)
    assert np.all(ch["contour"][0] == ds["contour"][1])
    assert np.all(ch["contour"][1] == ds["contour"][3])


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_feat_image():
    path = retrieve_data("fmt-hdf5_fl_2017.zip")
    ds = new_dataset(path)
    ds.filter.manual[0] = False
    ds.filter.manual[2] = False
    ch = new_dataset(ds)
    assert np.all(ch["image"][0] == ds["image"][1])
    assert np.all(ch["image"][1] == ds["image"][3])


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_feat_image_bg():
    path = retrieve_data("fmt-hdf5_fl_2017.zip")
    # add a fake image_bg column
    with h5py.File(path, mode="a") as h5:
        image_bg = h5["events"]["image"][:] // 2
        write(h5, data={"image_bg": image_bg}, mode="append")
    ds = new_dataset(path)
    ds.filter.manual[0] = False
    ds.filter.manual[2] = False
    ch = new_dataset(ds)
    assert np.all(ch["image_bg"][0] == ds["image_bg"][1])
    assert np.all(ch["image_bg"][1] == ds["image_bg"][3])


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_feat_mask():
    path = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    ds = new_dataset(path)
    ds.filter.manual[0] = False
    ds.filter.manual[2] = False
    ch = new_dataset(ds)
    assert np.all(ch["mask"][0] == ds["mask"][1])
    assert np.all(ch["mask"][1] == ds["mask"][3])


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_feat_trace():
    path = retrieve_data("fmt-hdf5_fl_2017.zip")
    ds = new_dataset(path)
    ds.filter.manual[0] = False
    ds.filter.manual[2] = False
    ch = new_dataset(ds)
    assert np.all(ch["trace"]["fl1_median"][0]
                  == ds["trace"]["fl1_median"][1])
    assert np.all(ch["trace"]["fl1_median"][1]
                  == ds["trace"]["fl1_median"][3])


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_features():
    path = retrieve_data("fmt-hdf5_fl_2017.zip")
    ds = new_dataset(path)
    ch = new_dataset(ds)
    assert ds.features == ch.features
    assert ds.features_innate == ch.features_innate
    assert ds.features_loaded == ch.features_loaded
    assert ds.features_scalar == ch.features_scalar


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_features_loaded():
    path = retrieve_data("fmt-hdf5_fl_2017.zip")
    ds = new_dataset(path)
    ch = new_dataset(ds)
    assert "volume" in ch.features
    assert "volume" not in ch.features_loaded
    assert "volume" not in ch.features_innate
    # compute volume, now it should be loaded
    ch["volume"]
    assert "volume" in ch.features_loaded
    assert "volume" in ds.features_loaded


def test_hierarchy_from_tdms():
    pytest.importorskip("nptdms")
    tdms_path = retrieve_data("fmt-tdms_fl-image_2016.zip")
    ds1 = new_dataset(tdms_path)
    ds2 = new_dataset(ds1)

    ds1.filter.manual[0] = False
    ds2.apply_filter()
    assert ds2.filter.all.shape[0] == ds1.filter.all.shape[0] - 1
    assert ds2["area_um"][0] == ds1["area_um"][1]


def test_index_deep_contour():
    data = example_data_dict(42, keys=["area_um", "contour", "deform"])
    ds = new_dataset(data)
    ds.filter.manual[3] = False
    c1 = new_dataset(ds)
    c1.filter.manual[1] = False
    c2 = new_dataset(c1)
    assert np.all(c2["contour"][3] == ds["contour"][5])


@pytest.mark.parametrize("feat", ["image", "image_bg", "mask"])
@pytest.mark.parametrize("idxs", [slice(0, 3), np.arange(3),
                                  [0, 1, 2], [True, True, True, False]])
def test_index_slicing(feat, idxs):
    data = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    ds = new_dataset(data)
    ds.filter.manual[2] = False
    ch = new_dataset(ds)

    ds_feat = ds[feat][np.array([0, 1, 3])]
    ch_feat = ch[feat]

    assert np.all(ch_feat[idxs] == ds_feat)


@pytest.mark.parametrize("feat", ["image", "mask"])
@pytest.mark.parametrize("idxs", [slice(0, 3), np.arange(3),
                                  [0, 1, 2], [True, True, True, False]])
def test_index_slicing_tdms_fails(feat, idxs):
    """The tdms-file format does not support slice/array indexing"""
    pytest.importorskip("nptdms")
    data = retrieve_data("fmt-tdms_shapein-2.0.1-no-image_2017.zip")
    ds = new_dataset(data)
    ds.filter.manual[2] = False
    ch = new_dataset(ds)

    with pytest.raises(NotImplementedError, match="scalar integers"):
        ch[feat][idxs]


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
@pytest.mark.parametrize("idxs", [slice(0, 3), np.arange(3),
                                  [0, 1, 2], [True, True, True, False]])
def test_index_slicing_trace(idxs):
    data = retrieve_data("fmt-hdf5_fl_2017.zip")
    ds = new_dataset(data)
    ds.filter.manual[2] = False
    ch = new_dataset(ds)

    ds_feat = ds["trace"]["fl1_median"][np.array([0, 1, 3])]
    ch_feat = ch["trace"]["fl1_median"]

    assert np.all(ch_feat[idxs] == ds_feat)


def test_manual_exclude():
    data = example_data_dict(42, keys=["area_um", "deform"])
    p = new_dataset(data)
    c1 = new_dataset(p)
    c2 = new_dataset(c1)
    c3 = new_dataset(c2)
    c1.filter.manual[0] = False
    c2.apply_filter()
    c2.filter.manual[1] = False
    c3.apply_filter()

    # simple exclusion of few events
    assert len(c3) == len(p) - 2

    # removing same event in parent removes the event from the
    # child altogether, including the manual filter
    c3.filter.manual[0] = False
    c2.filter.manual[0] = False
    c3.apply_filter()
    assert np.alltrue(c3.filter.manual)

    # reinserting the event in the parent, retrieves back
    # the manual filter in the child
    c2.filter.manual[0] = True
    c3.apply_filter()
    assert not c3.filter.manual[0]


def test_manual_exclude_parent_changed():
    data = example_data_dict(42, keys=["area_um", "tilt"])
    p = new_dataset(data)
    p.filter.manual[4] = False
    c = new_dataset(p)
    c.filter.manual[5] = False
    c.apply_filter()
    p.config["filtering"]["tilt min"] = 0
    p.config["filtering"]["tilt max"] = .5
    p.apply_filter()
    assert np.sum(p.filter.all) == 21
    # size of child is directly determined from parent
    assert len(c) == 21
    # filters have not yet been updated
    assert len(c.filter.all) == 41
    assert c.filter.parent_changed
    # the initially excluded event
    assert c.filter.retrieve_manual_indices(c) == [6]

    # try to change the excluded events
    try:
        c.filter.apply_manual_indices(c, [1, 2])
    except fmt_hierarchy.HierarchyFilterError:
        pass
    else:
        assert False, "expected HierarchyFilterError"

    # this can be resolved by applying the filter
    c.apply_filter()
    c.filter.apply_manual_indices(c, [1, 2])
    assert c.filter.retrieve_manual_indices(c) == [1, 2]


def test_same_hash_different_identifier():
    pytest.importorskip("nptdms")
    tdms_path = retrieve_data("fmt-tdms_fl-image_2016.zip")
    ds1 = new_dataset(tdms_path)
    ds1.filter.manual[0] = False
    ch1 = new_dataset(ds1)
    ch2 = new_dataset(ds1)
    assert len(ch1) == len(ds1) - 1
    assert ch1.hash == ch2.hash
    assert ch1.identifier != ch2.identifier


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
