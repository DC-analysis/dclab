"""Test filter hierarchies"""
import numpy as np
import pytest

import dclab
from dclab import new_dataset
from dclab.rtdc_dataset import fmt_hierarchy, RTDCWriter

from helper_methods import example_data_dict, retrieve_data


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_access_without_apply_filter():
    ds = new_dataset(retrieve_data("fmt-hdf5_image-bg_2020.zip"))
    ds.filter.manual[0] = False
    ds.apply_filter()
    ch = new_dataset(ds)
    assert len(ch["contour"]) == 4
    assert np.allclose(ch["area_um"][0], 228.25221)
    assert ds["area_um"].ndim == 1  # important for matplotlib
    ds.filter.manual[1] = False
    ds.apply_filter()
    # In the future, accessing this object might yield an error or warning,
    # depending on how problematic this approach is for the dclab users.
    assert np.allclose(ch["area_um"][0], 228.25221)
    assert ch.filter.parent_changed
    ch.rejuvenate()
    assert np.allclose(ch["area_um"][0], 156.349)
    assert not ch.filter.parent_changed


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
                                "emodulus viscosity model": "herold-2017",
                                "emodulus medium": "CellCarrier",
                                "emodulus temperature": 23.0,
                                }
    ch = new_dataset(ds)
    assert np.allclose(ch["emodulus"], ds["emodulus"], equal_nan=True)
    ds.config["calculation"]["emodulus temperature"] = 24.0
    # the user still has to call `apply_filter` to update the config:
    assert not ch.config["calculation"]["emodulus temperature"] == 24.0
    # ...and the cache
    assert not np.allclose(ch["emodulus"], ds["emodulus"], equal_nan=True)
    ch.rejuvenate()
    assert ch.config["calculation"]["emodulus temperature"] == 24.0
    assert np.allclose(ch["emodulus"], ds["emodulus"], equal_nan=True)


@pytest.mark.filterwarnings('ignore::dclab.features.emodulus.'
                            + 'YoungsModulusLookupTableExceededWarning')
def test_config_calculation_does_not_work_in_child_issue_92():
    """As long as #92 is alive, this test should pass."""
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0.16
    ds.config["setup"]["channel width"] = 30
    ds.config["imaging"]["pixel size"] = .34
    ch = new_dataset(ds)
    ch.config["calculation"] = {"emodulus lut": "LE-2D-FEM-19",
                                "emodulus medium": "CellCarrier",
                                "emodulus temperature": 23.0,
                                "emodulus viscosity": 0.5
                                }
    assert "emodulus" not in ch
    with pytest.raises(
            KeyError,
            match="If you are attempting to access an ancillary feature"):
        ch["emodulus"]


def test_hierarchy_logs():
    """Since version 0.50.1, hierarchy children inherit logs"""
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip"))
    assert len(ds.logs)
    ch = dclab.new_dataset(ds)
    assert ch.logs
    assert ch.logs["dclab-compress"][0] == "{"


def test_discouraged_array_dunder_childndarray():
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip"))
    ds2 = new_dataset(ds)
    with pytest.warns(UserWarning, match="It may consume a lot of memory"):
        ds2["image"].__array__()
    with pytest.warns(UserWarning, match="It may consume a lot of memory"):
        np.array(ds2["image"])


def test_discouraged_array_dunder_chiltraceitem():
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip"))
    ds2 = new_dataset(ds)
    with pytest.warns(UserWarning, match="It may consume a lot of memory"):
        ds2["trace"]["fl1_raw"].__array__()
    with pytest.warns(UserWarning, match="It may consume a lot of memory"):
        np.array(ds2["trace"]["fl1_raw"])


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_dtype_contour():
    ds = new_dataset(retrieve_data("fmt-hdf5_mask-contour_2018.zip"))
    assert ds["contour"].dtype == np.uint16
    ds2 = new_dataset(ds)
    assert ds2["contour"].dtype == np.uint16


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_dtype_mask_image():
    ds = new_dataset(retrieve_data("fmt-hdf5_image-bg_2020.zip"))
    assert ds["mask"].dtype == bool
    assert ds["image"].dtype == np.uint8
    ds2 = new_dataset(ds)
    assert ds2["mask"].dtype == bool
    assert ds2["image"].dtype == np.uint8


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_dtype_trace():
    ds = new_dataset(retrieve_data("fmt-hdf5_fl-no-contour_2019.zip"))
    assert ds["trace"]["fl1_raw"].dtype == np.int16
    ds2 = new_dataset(ds)
    assert ds2["trace"]["fl1_raw"].dtype == np.int16


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_hierarchy_shape_contour():
    ds = new_dataset(retrieve_data("fmt-hdf5_image-bg_2020.zip"))
    assert ds["contour"].shape == (5, np.nan, 2)
    ds.filter.manual[0] = False
    ds.apply_filter()
    ch = new_dataset(ds)
    assert ch["contour"].shape == (4, np.nan, 2)
    assert len(ch["contour"]) == 4


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_hierarchy_shape_image():
    ds = new_dataset(retrieve_data("fmt-hdf5_image-bg_2020.zip"))
    ds.filter.manual[0] = False
    ch = new_dataset(ds)
    assert "image" in ch.features_innate
    assert len(ch["image"]) == 4
    assert ch["image"].shape == (4, 80, 250)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_hierarchy_shape_mask():
    ds = new_dataset(retrieve_data("fmt-hdf5_image-bg_2020.zip"))
    ds.filter.manual[0] = False
    ch = new_dataset(ds)
    assert "mask" in ch.features_innate
    assert len(ch["mask"]) == 4
    assert ch["mask"].shape == (4, 80, 250)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_hierarchy_shape_trace():
    ds = new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
    assert len(ds) == 7
    ds.filter.manual[0] = False
    ds.filter.manual[1] = False
    ch = new_dataset(ds)
    assert len(ch) == 5
    assert "trace" in ch.features_innate
    assert ch["trace"].shape == (6, 5, 177)
    assert ch["trace"]["fl1_raw"].shape == (5, 177)
    assert ch["trace"]["fl1_raw"][0].shape == (177,)
    assert len(ch["trace"]) == 6
    assert len(ch["trace"]["fl1_raw"]) == 5
    assert len(ch["trace"]["fl1_raw"][0]) == 177


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
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
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
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
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
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
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_feat_image_bg():
    path = retrieve_data("fmt-hdf5_fl_2017.zip")
    # add a fake image_bg column
    with RTDCWriter(path) as hw:
        image_bg = hw.h5file["events"]["image"][:] // 2
        hw.store_feature("image_bg", image_bg)
    ds = new_dataset(path)
    ds.filter.manual[0] = False
    ds.filter.manual[2] = False
    ch = new_dataset(ds)
    assert np.all(ch["image_bg"][0] == ds["image_bg"][1])
    assert np.all(ch["image_bg"][1] == ds["image_bg"][3])


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
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
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
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
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
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
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_feature_contained():
    pytest.importorskip("nptdms")
    path_im = retrieve_data("fmt-tdms_fl-image_2016.zip")
    ds_im = new_dataset(path_im)
    path_no_im = retrieve_data("fmt-tdms_2fl-no-image_2017.zip")
    ds_no_im = new_dataset(path_no_im)

    assert "image" in ds_im
    assert "mask" in ds_im
    assert "image" not in ds_no_im
    assert "mask" not in ds_no_im


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
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


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_hierarchy_from_tdms():
    pytest.importorskip("nptdms")
    tdms_path = retrieve_data("fmt-tdms_fl-image_2016.zip")
    ds1 = new_dataset(tdms_path)
    ds2 = new_dataset(ds1)

    ds1.filter.manual[0] = False
    ds2.rejuvenate()
    assert ds2.filter.all.shape[0] == ds1.filter.all.shape[0] - 1
    assert ds2["area_um"][0] == ds1["area_um"][1]


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_hierarchy_ufuncs():
    path = retrieve_data("fmt-hdf5_fl_2018.zip")

    ds = new_dataset(path)
    ds.filter.manual[0] = False
    ds.apply_filter()
    ch = new_dataset(ds)

    assert len(ds) == 7
    assert len(ch) == 6

    # reference
    assert np.min(ds["area_cvx"]) == 226.0
    assert np.max(ds["area_cvx"]) == 287.5
    assert np.allclose(np.mean(ds["area_cvx"]), 255.28572, rtol=0, atol=1e-5)

    # filtered
    assert np.min(ch["area_cvx"]) == 226.0
    assert np.max(ch["area_cvx"]) == 287.5
    assert np.allclose(np.mean(ch["area_cvx"]), 256.0, rtol=0, atol=1e-5)

    # change the filter and make sure things changed for the child
    ds.filter.manual[:4] = False
    ch.rejuvenate()

    assert np.min(ch["area_cvx"]) == 226.0
    assert np.max(ch["area_cvx"]) == 279.5
    assert np.allclose(np.mean(ch["area_cvx"]), 249.83333, rtol=0, atol=1e-5)

    # make sure ufuncs are actually used
    assert ch["area_cvx"]._ufunc_attrs["min"] == 226.0


def test_index_deep_contour():
    data = example_data_dict(42, keys=["area_um", "contour", "deform"])
    ds = new_dataset(data)
    ds.filter.manual[3] = False
    c1 = new_dataset(ds)
    c1.filter.manual[1] = False
    c2 = new_dataset(c1)
    assert np.all(c2["contour"][3] == ds["contour"][5])


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
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
                                  [0, 1, 2], [False]+42*[True]])
def test_index_slicing_tdms_fails(feat, idxs):
    """The tdms-file format does not support slice/array indexing"""
    pytest.importorskip("nptdms")
    data = retrieve_data("fmt-tdms_fl-image_2016.zip")
    ds = new_dataset(data)
    ds.filter.manual[2] = False
    ch = new_dataset(ds)

    with pytest.raises(NotImplementedError, match="scalar integers"):
        ch[feat][idxs]


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
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
    c2.rejuvenate()
    c2.filter.manual[1] = False
    c3.rejuvenate()

    # simple exclusion of few events
    assert len(c3) == len(p) - 2

    # removing same event in parent removes the event from the
    # child altogether, including the manual filter
    c3.filter.manual[0] = False
    c2.filter.manual[0] = False
    c3.rejuvenate()
    assert np.all(c3.filter.manual)

    # reinserting the event in the parent, retrieves back
    # the manual filter in the child
    c2.filter.manual[0] = True
    c3.rejuvenate()
    assert not c3.filter.manual[0]


def test_manual_exclude_parent_changed():
    data = example_data_dict(42, keys=["area_um", "tilt"])
    p = new_dataset(data)
    p.filter.manual[4] = False
    c = new_dataset(p)
    c.filter.manual[5] = False
    c.rejuvenate()
    p.config["filtering"]["tilt min"] = 0
    p.config["filtering"]["tilt max"] = .5
    p.apply_filter()
    assert np.sum(p.filter.all) == 21
    # size of child is not directly determined from parent
    assert len(c) == 41
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
    c.rejuvenate()
    c.filter.apply_manual_indices(c, [1, 2])
    assert c.filter.retrieve_manual_indices(c) == [1, 2]


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
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
