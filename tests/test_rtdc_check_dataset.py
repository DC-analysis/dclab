import sys

import h5py
import numpy as np
import pytest

import dclab.rtdc_dataset.config
from dclab.rtdc_dataset import check, check_dataset, fmt_tdms, new_dataset, \
    RTDCWriter

from helper_methods import example_data_dict, retrieve_data


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_basic():
    h5path = retrieve_data("fmt-hdf5_fl_2017.zip")
    viol, aler, info = check_dataset(h5path)
    # Features: Unknown key 'ncells'
    # Metadata: Mismatch [imaging] 'roi size x' and feature image (50 vs 90)
    # Metadata: Mismatch [imaging] 'roi size y' and feature image (90 vs 50)
    # Metadata: Missing key [fluorescence] 'channels installed'
    # Metadata: Missing key [fluorescence] 'laser count'
    # Metadata: Missing key [fluorescence] 'lasers installed'
    # Metadata: Missing key [fluorescence] 'samples per event'
    # Metadata: fluorescence channel count inconsistent
    assert len(viol) == 8
    # "HDF5: '/image': attribute 'CLASS' should be fixed-length ASCII string",
    # "HDF5: '/image': attribute 'IMAGE_SUBCLASS' should be fixed-length ...
    # "HDF5: '/image': attribute 'IMAGE_VERSION' should be fixed-length ...
    # "Metadata: Flow rates don't add up (sh 0.6 + sam 0.1 != channel 0.16)",
    # "Metadata: Flow rates don't add up (sh 0.6 + sam 0.1 != channel 0.16)",
    # "Metadata: Flow rates don't add up (sh 0.6 + sam 0.1 != channel 0.16)",
    # "Metadata: Missing key [fluorescence] 'channel 1 name'",
    # "Metadata: Missing key [fluorescence] 'channel 2 name'",
    # "Metadata: Missing key [fluorescence] 'channel 3 name'",
    # "Metadata: Missing key [setup] 'identifier'",
    # "Metadata: Missing section 'online_contour'"
    # "UnknownConfigurationKeyWarning: Unknown key 'exposure time' ...",
    # "UnknownConfigurationKeyWarning: Unknown key 'flash current' ...",
    # "UserWarning: Type of confguration key [fluorescence]: sample rate ...",
    # "UserWarning: Type of confguration key [imaging]: roi position x ...",
    # "UserWarning: Type of confguration key [imaging]: roi position y ..."]
    assert len(aler) == 16
    assert "Data file format: hdf5" in info
    assert "Fluorescence: True" in info
    assert "Compression: None" in info


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_complete():
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    viol, aler, info = check_dataset(h5path)
    assert len(viol) == 0
    # [fluorescence]: sample rate should be <class 'numbers.Integral'>
    # [imaging]: roi position x should be <class 'numbers.Integral'>
    # [imaging]: roi position y should be <class 'numbers.Integral'>
    assert len(aler) == 3
    assert "Data file format: hdf5" in info
    assert "Fluorescence: True" in info


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_complete_user_metadata():
    """Setting any user metadata is allowed"""
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    metadata = {"channel area": 100.5,
                "inlet": True,
                "n_constrictions": 3,
                "channel information": "other information"}
    with new_dataset(h5path) as ds:
        ds.config.update({"user": metadata})
        expath = h5path.with_name("exported.rtdc")
        ds.export.hdf5(expath, features=ds.features_innate)
        viol, aler, info = check_dataset(ds)
        assert len(viol) == 0
        assert len(aler) == 0
        assert "Data file format: hdf5" in info
        assert "Fluorescence: True" in info


def test_exact():
    pytest.importorskip("nptdms")
    h5path = retrieve_data("fmt-tdms_2fl-no-image_2017.zip")
    viol, aler, info = check_dataset(h5path)
    known_viol = [
        "Metadata: Missing key [fluorescence] 'channel count'",
        "Metadata: Missing key [fluorescence] 'channels installed'",
        "Metadata: Missing key [fluorescence] 'laser count'",
        "Metadata: Missing key [fluorescence] 'lasers installed'",
        "Metadata: Missing key [fluorescence] 'samples per event'",
    ]
    known_aler = [
        "Metadata: Missing key [fluorescence] 'channel 1 name'",
        "Metadata: Missing key [fluorescence] 'channel 2 name'",
        "Metadata: Missing key [online_contour] 'no absdiff'",
        "Metadata: Missing key [setup] 'identifier'",
        "Metadata: Missing key [setup] 'module composition'",
        "Negative value for feature(s): fl2_max",
    ]
    known_info = [
        'Compression: None',
        'Data file format: tdms',
        'Fluorescence: True',
    ]
    assert set(viol) == set(known_viol)
    assert set(aler) == set(known_aler)
    assert set(info) == set(known_info)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_icue():
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with check.IntegrityChecker(h5path) as ic:
        cues = ic.check()
    assert cues[0] != cues[1]
    levels = check.ICue.get_level_summary(cues)
    assert levels["info"] >= 2
    # [fluorescence]: sample rate should be <class 'numbers.Integral'>
    # [imaging]: roi position x should be <class 'numbers.Integral'>
    # [imaging]: roi position y should be <class 'numbers.Integral'>
    assert levels["alert"] == 3
    assert levels["violation"] == 0
    assert cues[0].msg in cues[0].__repr__()


def test_ic_expand_section():
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ds1 = new_dataset(ddict)
    ds2 = new_dataset(ddict)
    with check.IntegrityChecker(ds1) as ic:
        cues1 = ic.check_metadata_missing(expand_section=True)
    with check.IntegrityChecker(ds2) as ic:
        cues2 = ic.check_metadata_missing(expand_section=False)
    assert len(cues1) > len(cues2)


def test_ic_feature_size_scalar():
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ddict["bright_sd"] = np.linspace(10, 20, 1000)
    ds = new_dataset(ddict)
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_feature_size()
    for cue in cues:
        if cue.category == "feature size":
            assert cue.msg == "Features: wrong event count: 'bright_sd' " \
                              + "(1000 of 8472)"
            break
    else:
        assert False


def test_ic_feature_size_trace():
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ddict["trace"] = {"fl1_raw": [range(10)] * 1000}
    ds = new_dataset(ddict)
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_feature_size()
    for cue in cues:
        if cue.category == "feature size":
            assert cue.msg == "Features: wrong event count: 'trace/fl1_raw'" \
                              + " (1000 of 8472)"
            break
    else:
        assert False


def test_ic_fl_metadata_channel_names():
    ddict = example_data_dict(size=8472, keys=["area_um", "deform", "fl1_max"])
    ddict["trace"] = {"fl1_raw": [range(10)] * 1000}
    ds = new_dataset(ddict)
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_fl_metadata_channel_names()
    assert cues[0].category == "metadata missing"
    assert cues[0].cfg_section == "fluorescence"
    assert cues[0].cfg_key == "channel 1 name"


def test_ic_fl_metadata_channel_names_2():
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    ds.config["fluorescence"]["channel 1 name"] = "peter"
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_fl_metadata_channel_names()
    assert cues[0].category == "metadata invalid"
    assert cues[0].cfg_section == "fluorescence"
    assert cues[0].cfg_key == "channel 1 name"


def test_ic_fl_num_channels():
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    ds.config["fluorescence"]["channel count"] = 3
    ds.config["fluorescence"]["channel 1 name"] = "hans"
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_fl_num_channels()
    assert cues[0].category == "metadata wrong"
    assert cues[0].cfg_section == "fluorescence"
    assert cues[0].cfg_key == "channel count"


def test_ic_fl_num_lasers():
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    ds.config["fluorescence"]["laser count"] = 3
    ds.config["fluorescence"]["laser 1 lambda"] = 550
    ds.config["fluorescence"]["laser 1 power"] = 20
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_fl_num_lasers()
    assert cues[0].category == "metadata wrong"
    assert cues[0].cfg_section == "fluorescence"
    assert cues[0].cfg_key == "laser count"


def test_ic_flow_rate_not_zero():
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    ds.config["setup"]["flow rate"] = 0
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_metadata_bad_greater_zero()
    assert cues[0].category == "metadata wrong"
    assert cues[0].cfg_section == "setup"
    assert cues[0].cfg_key == "flow rate"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_ic_fmt_hdf5_image1():
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with h5py.File(h5path, "a") as h5:
        h5["events/image"].attrs["CLASS"] = "bad"
    with check.IntegrityChecker(h5path) as ic:
        cues = ic.check_fmt_hdf5()
    assert cues[0].category == "format HDF5"
    assert cues[0].msg == "HDF5: '/image': attribute 'CLASS' should be " \
                          + "fixed-length ASCII string"
    assert cues[0].level == "alert"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.skipif(sys.version_info < (3, 0),
                    reason="requires python3 or higher")
def test_ic_fmt_hdf5_image2():
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with h5py.File(h5path, "a") as h5:
        h5["events/image"].attrs["CLASS"] = np.string_("bad")
    with check.IntegrityChecker(h5path) as ic:
        cues = ic.check_fmt_hdf5()
    assert cues[0].category == "format HDF5"
    assert cues[0].msg == "HDF5: '/image': attribute 'CLASS' should have " \
                          + "value 'b'IMAGE''"
    assert cues[0].level == "alert"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_ic_fmt_hdf5_image3():
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with h5py.File(h5path, "a") as h5:
        del h5["events/image"].attrs["CLASS"]
    with check.IntegrityChecker(h5path) as ic:
        cues = ic.check_fmt_hdf5()
    assert cues[0].category == "format HDF5"
    assert cues[0].msg == "HDF5: '/image': missing attribute 'CLASS'"
    assert cues[0].level == "alert"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_ic_fmt_hdf5_image_bg():
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    # add a fake image_bg column
    with h5py.File(h5path, "a") as h5:
        image_bg = h5["events"]["image"][:] // 2
        hw = RTDCWriter(h5)
        hw.store_feature("image_bg", image_bg)
        del h5["events/image_bg"].attrs["CLASS"]
    with check.IntegrityChecker(h5path) as ic:
        cues = ic.check_fmt_hdf5()
    assert cues[0].category == "format HDF5"
    assert cues[0].msg == "HDF5: '/image_bg': missing attribute 'CLASS'"
    assert cues[0].level == "alert"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_ic_fmt_hdf5_logs():
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    hw = RTDCWriter(h5path)
    hw.store_log("test", ["asdasd"*100])
    hw.store_log("M1_para.ini", ["asdasd"*100])
    with check.IntegrityChecker(h5path) as ic:
        cues = ic.check_fmt_hdf5()
    assert len(cues) == 1
    assert cues[0].category == "format HDF5"
    assert cues[0].msg == 'Logs: test line 0 exceeds maximum line length 100'
    assert cues[0].level == "alert"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_ic_metadata_bad():
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    # Since version 0.35, metadata are checked in `Configuration` class
    with pytest.warns(dclab.rtdc_dataset.config.WrongConfigurationTypeWarning,
                      match="run index"):
        ds.config["experiment"]["run index"] = "1"
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_metadata_bad()
    assert len(cues) == 0


def test_ic_metadata_choices_medium():
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    ds.config["setup"]["medium"] = "honey"
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_metadata_choices()
    # changed in 0.29.1: medium can now be an arbitrary string
    # except for an empty string.
    assert len(cues) == 0


def test_check_metadata_hdf5_type_issue_139():
    """Check that chip region is lower-case"""
    h5path = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    with h5py.File(h5path, "a") as h5:
        h5.attrs["setup:chip region"] = "Channel"
    with check.IntegrityChecker(h5path) as ic:
        cues = ic.check_metadata_hdf5_type()
    assert len(cues) == 1
    assert cues[0].msg.count("channel")
    assert cues[0].msg.count("Channel")


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.EmptyConfigurationKeyWarning')
def test_ic_metadata_empty_string():
    """Empty metadata values are ignored with a warning in dclab>0.33.2"""
    path = retrieve_data("fmt-hdf5_fl_2018.zip")
    # add empty attribute
    with h5py.File(path, "r+") as h5:
        h5.attrs["setup:medium"] = ""
    ds = new_dataset(path)
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_metadata_missing()
    assert cues[0].category == "metadata missing"
    assert cues[0].level == "violation"
    assert cues[0].cfg_section == "setup"
    assert cues[0].cfg_key == "medium"


def test_ic_fl_max():
    # Testing dataset with negative fl_max values
    ddict = example_data_dict(size=8472, keys=["fl1_max"])
    ddict["fl1_max"] -= min(ddict["fl1_max"]) + 1
    ds = new_dataset(ddict)
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_fl_max_positive()
    assert cues[0].level == "alert"
    assert cues[0].category == "feature data"

    # Testing dataset with fl_max values of 0.1
    ddict = example_data_dict(size=8472, keys=["fl1_max"])
    ddict["fl1_max"] -= min(ddict["fl1_max"]) - 0.1
    ds = new_dataset(ddict)
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_fl_max_positive()
    assert cues[0].level == "alert"
    assert cues[0].category == "feature data"

    # Testing dataset with fl_max values > 0.1
    ddict = example_data_dict(size=8472, keys=["fl1_max"])
    ddict["fl1_max"] -= min(ddict["fl1_max"]) - 1
    ds = new_dataset(ddict)
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_fl_max_positive()
    assert not cues


def test_ic_fl_max_ctc():
    # Testing dataset with negative fl_max_ctc values
    ddict = example_data_dict(size=8472, keys=["fl1_max_ctc"])
    ddict["fl1_max_ctc"] -= min(ddict["fl1_max_ctc"]) + 1
    ds = new_dataset(ddict)
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_fl_max_ctc_positive()
    assert cues[0].level == "alert"
    assert cues[0].category == "feature data"

    # Testing dataset with fl_max_ctc values of 0.1
    ddict = example_data_dict(size=8472, keys=["fl1_max_ctc"])
    ddict["fl1_max_ctc"] -= min(ddict["fl1_max_ctc"]) - 0.1
    ds = new_dataset(ddict)
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_fl_max_ctc_positive()
    assert cues[0].level == "alert"
    assert cues[0].category == "feature data"

    # Testing dataset with fl_max_ctc values > 0.1
    ddict = example_data_dict(size=8472, keys=["fl1_max_ctc"])
    ddict["fl1_max_ctc"] -= min(ddict["fl1_max_ctc"]) - 1
    ds = new_dataset(ddict)
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_fl_max_ctc_positive()
    assert not cues


def test_ic_invalid_dataset():
    # Testing if IC throws NotImplementedError for hierarchy datasets
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    ds_child = new_dataset(ds)
    with check.IntegrityChecker(ds_child) as ic:
        with pytest.raises(NotImplementedError):
            ic.check()

    # Testing if IC throws NotImplementedError for raw-datasets with
    # applied filters
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    ds.config["filtering"]["area_um max"] = 100
    ds.config["filtering"]["area_um min"] = 1
    ds.apply_filter()
    with check.IntegrityChecker(ds) as ic:
        with pytest.raises(NotImplementedError):
            ic.check()


def test_ic_sanity():
    h5path = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    with h5py.File(h5path, "a") as h5:
        del h5["events"]["deform"]
        h5["events"]["deform"] = np.ones(100) * .1
    with check.IntegrityChecker(h5path) as ic:
        cues = ic.sanity_check()
    assert len(cues) == 1
    assert cues[0].category == "feature size"
    assert cues[0].msg.count("Sanity check failed:")
    assert cues[0].msg.count("deform")
    assert cues[0].level == "violation"


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'feat_anc_core.ancillary_feature.'
                            + 'BadFeatureSizeWarning')
@pytest.mark.skipif(sys.version_info < (3, 6),
                    reason="requires python3.6 or higher")
def test_invalid_medium():
    pytest.importorskip("nptdms")
    h5path = retrieve_data("fmt-tdms_minimal_2016.zip")
    para = h5path.with_name("M1_para.ini")
    cfg = para.read_text().split("\n")
    cfg.insert(3, "Buffer Medium = unknown_bad!")
    para.write_text("\n".join(cfg))
    viol, _, _ = check_dataset(h5path)
    # changed in 0.29.1: medium can now be an arbitrary string
    # except for an empty string.
    assert "Metadata: Invalid value [setup] medium: 'unknown_bad!'" not in viol


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'feat_anc_core.ancillary_feature.'
                            + 'BadFeatureSizeWarning')
def test_load_with():
    pytest.importorskip("nptdms")
    h5path = retrieve_data("fmt-tdms_minimal_2016.zip")
    known_aler = [
        "Metadata: Missing key [setup] 'flow rate sample'",
        "Metadata: Missing key [setup] 'flow rate sheath'",
        "Metadata: Missing key [setup] 'identifier'",
        "Metadata: Missing key [setup] 'module composition'",
        "Metadata: Missing key [setup] 'software version'",
    ]
    known_viol = [
        "Features: wrong event count: 'contour' (14 of 156)",
        "Features: wrong event count: 'mask' (14 of 156)",
        "Metadata: Missing key [setup] 'medium'",
    ]
    with new_dataset(h5path) as ds:
        viol, aler, _ = check_dataset(ds)
        assert set(viol) == set(known_viol)
        assert set(aler) == set(known_aler)


def test_missing_file():
    pytest.importorskip("nptdms")
    h5path = retrieve_data("fmt-tdms_2fl-no-image_2017.zip")
    h5path.with_name("M1_para.ini").unlink()
    try:
        check_dataset(h5path)
    except fmt_tdms.IncompleteTDMSFileFormatError:
        pass
    else:
        assert False


def test_ml_class():
    """Test score data outside boundary"""
    data = {"ml_score_001": [.1, 10, -10, 0.01, .89],
            "ml_score_002": [.2, .1, .4, 0, .4],
            }
    ds = new_dataset(data)
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_ml_class()
        assert len(cues) == 1
        assert "ml_score_001" in cues[0].msg


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'feat_anc_core.ancillary_feature.'
                            + 'BadFeatureSizeWarning')
def test_no_fluorescence():
    pytest.importorskip("nptdms")
    h5path = retrieve_data("fmt-tdms_minimal_2016.zip")
    _, _, info = check_dataset(h5path)
    known_info = [
        'Compression: None',
        'Data file format: tdms',
        'Fluorescence: False',
    ]
    assert set(info) == set(known_info)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_online_polygon_filters():
    """Shape-In 2.3 supports online polygon filters"""
    path = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # add an artificial online polygon filter
    with h5py.File(path, "a") as h5:
        # set soft filter to True
        h5.attrs["online_filter:area_um,deform soft limit"] = True
        # set filter values
        pf_name = "online_filter:area_um,deform polygon points"
        area_um = h5["events"]["area_um"]
        deform = h5["events"]["deform"]
        pf_points = np.array([
            [np.mean(area_um) + np.std(area_um),
             np.mean(deform)],
            [np.mean(area_um) + np.std(area_um),
             np.mean(deform) + np.std(deform)],
            [np.mean(area_um),
             np.mean(deform) + np.std(deform)],
        ])
        h5.attrs[pf_name] = pf_points

    # see if we can open the file without any error
    with check.IntegrityChecker(path) as ic:
        cues = [cc for cc in ic.check() if cc.level != "info"]
        # [imaging]: roi position x should be <class 'numbers.Integral'>
        # [imaging]: roi position y should be <class 'numbers.Integral'>
        assert len(cues) == 2


def test_online_polygon_filters_real_data():
    """Shape-In 2.3 supports online polygon filters"""
    path = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")

    # see if we can open the file without any error
    with check.IntegrityChecker(path) as ic:
        cues = [cc for cc in ic.check() if cc.level != "info"]
        assert len(cues) == 0


@pytest.mark.parametrize("shape", [[3, 3], [2, 2], [1, 2], [10, 3]])
def test_online_polygon_filters_wrong_shape(shape):
    """Shape-In 2.3 supports online polygon filters (test for shape)"""
    path = retrieve_data("fmt-hdf5_mask-contour_2018.zip")

    # Since 0.35.0 we really check the configuration key types.
    # Just make sure that they are properly set:
    with h5py.File(path, "a") as h5:
        for key in ["imaging:roi position x",
                    "imaging:roi position y"]:
            h5.attrs[key] = int(h5.attrs[key])

    # add an artificial online polygon filter
    with h5py.File(path, "a") as h5:
        # set soft filter to True
        h5.attrs["online_filter:area_um,deform soft limit"] = True
        # set filter values
        pf_name = "online_filter:area_um,deform polygon points"
        pf_points = np.arange(shape[0]*shape[1]).reshape(*shape)
        h5.attrs[pf_name] = pf_points

    # see if we can open the file without any error
    with check.IntegrityChecker(path) as ic:
        cues = [cc for cc in ic.check() if cc.level != "info"]
        assert len(cues) == 1
        assert cues[0].category == "metadata wrong"
        assert cues[0].level == "violation"
        assert cues[0].cfg_section == "online_filter"
        assert cues[0].cfg_key == "area_um,deform polygon points"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.parametrize("si_version", ["2.2.1.0", "2.2.1.0dev", "2.2.2.0dev",
                                        "2.2.2.0", "2.2.2.1", "2.2.2.2",
                                        "2.2.2.4", "2.2.2.4", "2.3.0.0"])
def test_shapein_issue3_bad_medium(si_version):
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with h5py.File(h5path, "a") as h5:
        h5.attrs["setup:software version"] = si_version
        h5.attrs["setup:medium"] = "CellCarrierB"
    ds = new_dataset(h5path)
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_shapein_issue3_bad_medium()
        assert len(cues) == 1
        assert cues[0].cfg_key == "medium"
        assert cues[0].cfg_section == "setup"
        assert cues[0].category == "metadata wrong"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.parametrize("si_version", ["2.2.0.3", "2.3.1.0"])
def test_shapein_issue3_bad_medium_control(si_version):
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with h5py.File(h5path, "a") as h5:
        h5.attrs["setup:software version"] = si_version
        h5.attrs["setup:medium"] = "CellCarrierB"
    ds = new_dataset(h5path)
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_shapein_issue3_bad_medium()
        assert len(cues) == 0


def test_temperature():
    # there are probably a million things wrong with this dataset, but
    # we are only looking for the temperature thing
    ddict = example_data_dict(size=8472, keys=["area_um", "deform", "temp"])
    ds = new_dataset(ddict)
    sstr = "Metadata: Missing key [setup] 'temperature', " \
           + "because the 'temp' feature is given"
    _, aler, _ = check_dataset(ds)
    assert sstr in aler


def test_wrong_samples_per_event():
    pytest.importorskip("nptdms")
    h5path = retrieve_data("fmt-tdms_2fl-no-image_2017.zip")
    with h5path.with_name("M1_para.ini").open("a") as fd:
        fd.write("Samples Per Event = 10\n")
    msg = "Metadata: wrong number of samples per event: fl1_median " \
          + "(expected 10, got 566)"
    viol, _, _ = check_dataset(h5path)
    assert msg in viol


if __name__ == "__main__":
    # Run all tests
    _loc = locals()
    for _key in list(_loc.keys()):
        if _key.startswith("test_") and hasattr(_loc[_key], "__call__"):
            _loc[_key]()
