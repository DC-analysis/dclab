#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals

import sys

import h5py
import numpy as np
import pytest

from dclab.rtdc_dataset import check, check_dataset, fmt_tdms, new_dataset, \
    write

from helper_methods import cleanup, example_data_dict, retrieve_data


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.config.UnknownConfigurationKeyWarning')
def test_basic():
    h5path = retrieve_data("rtdc_data_hdf5_contour_image_trace.zip")
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
    assert len(aler) == 13
    assert "Data file format: hdf5" in info
    assert "Fluorescence: True" in info
    assert "Compression: Partial (1 of 25)" in info
    cleanup()


def test_complete():
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    viol, aler, info = check_dataset(h5path)
    assert len(viol) == 0
    assert len(aler) == 0
    assert "Data file format: hdf5" in info
    assert "Fluorescence: True" in info
    cleanup()


def test_exact():
    h5path = retrieve_data("rtdc_data_traces_2flchan.zip")
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
    ]
    known_info = [
        'Compression: None',
        'Data file format: tdms',
        'Fluorescence: True',
    ]
    assert set(viol) == set(known_viol)
    assert set(aler) == set(known_aler)
    assert set(info) == set(known_info)
    cleanup()


def test_icue():
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    with check.IntegrityChecker(h5path) as ic:
        cues = ic.check()
    assert cues[0] != cues[1]
    levels = check.ICue.get_level_summary(cues)
    assert levels["info"] >= 2
    assert levels["alert"] == 0
    assert levels["violation"] == 0
    assert cues[0].msg in cues[0].__repr__()
    cleanup()


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


def test_ic_fmt_hdf5_image1():
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    with h5py.File(h5path, "a") as h5:
        h5["events/image"].attrs["CLASS"] = "bad"
    with check.IntegrityChecker(h5path) as ic:
        cues = ic.check_fmt_hdf5()
    assert cues[0].category == "format HDF5"
    assert cues[0].msg == "HDF5: '/image': attribute 'CLASS' should be " \
                          + "fixed-length ASCII string"
    assert cues[0].level == "alert"
    cleanup()


@pytest.mark.skipif(sys.version_info < (3, 0),
                    reason="requires python3 or higher")
def test_ic_fmt_hdf5_image2():
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    with h5py.File(h5path, "a") as h5:
        h5["events/image"].attrs["CLASS"] = np.string_("bad")
    with check.IntegrityChecker(h5path) as ic:
        cues = ic.check_fmt_hdf5()
    assert cues[0].category == "format HDF5"
    assert cues[0].msg == "HDF5: '/image': attribute 'CLASS' should have " \
                          + "value 'b'IMAGE''"
    assert cues[0].level == "alert"
    cleanup()


def test_ic_fmt_hdf5_image3():
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    with h5py.File(h5path, "a") as h5:
        del h5["events/image"].attrs["CLASS"]
    with check.IntegrityChecker(h5path) as ic:
        cues = ic.check_fmt_hdf5()
    assert cues[0].category == "format HDF5"
    assert cues[0].msg == "HDF5: '/image': missing attribute 'CLASS'"
    assert cues[0].level == "alert"
    cleanup()


def test_ic_fmt_hdf5_logs():
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    write(h5path, logs={
        "test": ["asdasd"*100],
        "M1_para.ini":  ["asdasd"*100],  # should be ignored
    }, mode="append")
    with check.IntegrityChecker(h5path) as ic:
        cues = ic.check_fmt_hdf5()
    assert len(cues) == 1
    assert cues[0].category == "format HDF5"
    assert cues[0].msg == 'Logs: test line 0 exceeds maximum line length 100'
    assert cues[0].level == "alert"
    cleanup()


def test_ic_metadata_bad():
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    ds.config["experiment"]["run index"] = "1"
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_metadata_bad()
    assert cues[0].category == "metadata dtype"
    assert cues[0].level == "violation"
    assert cues[0].cfg_section == "experiment"
    assert cues[0].cfg_key == "run index"


def test_ic_metadata_choices():
    ddict = example_data_dict(size=8472, keys=["area_um", "deform"])
    ds = new_dataset(ddict)
    ds.config["setup"]["medium"] = "honey"
    with check.IntegrityChecker(ds) as ic:
        cues = ic.check_metadata_choices()
    assert cues[0].category == "metadata wrong"
    assert cues[0].level == "violation"
    assert cues[0].cfg_section == "setup"
    assert cues[0].cfg_key == "medium"


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'ancillaries.ancillary_feature.'
                            + 'BadFeatureSizeWarning')
@pytest.mark.skipif(sys.version_info < (3, 6),
                    reason="requires python3.6 or higher")
def test_invalid_medium():
    h5path = retrieve_data("rtdc_data_minimal.zip")
    para = h5path.with_name("M1_para.ini")
    cfg = para.read_text().split("\n")
    cfg.insert(3, "Buffer Medium = unknown_bad!")
    para.write_text("\n".join(cfg))
    viol, _, _ = check_dataset(h5path)
    assert "Metadata: Invalid value [setup] medium: 'unknown_bad!'" in viol
    cleanup()


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'ancillaries.ancillary_feature.'
                            + 'BadFeatureSizeWarning')
def test_load_with():
    h5path = retrieve_data("rtdc_data_minimal.zip")
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
    cleanup()


def test_missing_file():
    h5path = retrieve_data("rtdc_data_traces_2flchan.zip")
    h5path.with_name("M1_para.ini").unlink()
    try:
        check_dataset(h5path)
    except fmt_tdms.IncompleteTDMSFileFormatError:
        pass
    else:
        assert False
    cleanup()


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
                            + 'ancillaries.ancillary_feature.'
                            + 'BadFeatureSizeWarning')
def test_no_fluorescence():
    h5path = retrieve_data("rtdc_data_minimal.zip")
    _, _, info = check_dataset(h5path)
    known_info = [
        'Compression: None',
        'Data file format: tdms',
        'Fluorescence: False',
    ]
    assert set(info) == set(known_info)
    cleanup()


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
    h5path = retrieve_data("rtdc_data_traces_2flchan.zip")
    with h5path.with_name("M1_para.ini").open("a") as fd:
        fd.write("Samples Per Event = 10\n")
    msg = "Metadata: wrong number of samples per event: fl1_median " \
          + "(expected 10, got 566)"
    viol, _, _ = check_dataset(h5path)
    assert msg in viol
    cleanup()


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
