#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals

import sys

import pytest

from dclab.rtdc_dataset import check_dataset, fmt_tdms, new_dataset

from helper_methods import cleanup, retrieve_data


@pytest.mark.filterwarnings(
    'ignore::dclab.rtdc_dataset.fmt_hdf5.UnknownKeyWarning')
def test_basic():
    h5path = retrieve_data("rtdc_data_hdf5_contour_image_trace.zip")
    viol, aler, info = check_dataset(h5path)
    # Features: Unknown key 'ncells'
    # Metadata: Missing key [fluorescence] 'channels installed'
    # Metadata: Missing key [fluorescence] 'laser count'
    # Metadata: Missing key [fluorescence] 'lasers installed'
    # Metadata: Missing key [fluorescence] 'samples per event'
    # Metadata: Unknown key [imaging] 'exposure time'
    # Metadata: Unknown key [imaging] 'flash current'
    # Metadata: Unknown key [setup] 'temperature'
    # Metadata: fluorescence channel count inconsistent
    assert len(viol) == 9
    # "HDF5: '/image': attribute 'CLASS' should be fixed-length ASCII string",
    # "HDF5: '/image': attribute 'IMAGE_SUBCLASS' should be fixed-length ...
    # "HDF5: '/image': attribute 'IMAGE_VERSION' should be fixed-length ...
    # "Metadata: Missing key [fluorescence] 'channel 1 name'",
    # "Metadata: Missing key [fluorescence] 'channel 2 name'",
    # "Metadata: Missing key [fluorescence] 'channel 3 name'",
    # "Metadata: Missing key [setup] 'identifier'",
    # "Metadata: Missing section 'online_contour'"
    assert len(aler) == 8
    assert "Data file format: hdf5" in info
    assert "Fluorescence: True" in info
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
    known_info = ['Data file format: tdms', 'Fluorescence: True']
    assert set(viol) == set(known_viol)
    assert set(aler) == set(known_aler)
    assert set(info) == set(known_info)
    cleanup()


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


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'ancillaries.ancillary_feature.'
                            + 'BadFeatureSizeWarning')
def test_no_fluorescence():
    h5path = retrieve_data("rtdc_data_minimal.zip")
    _, _, info = check_dataset(h5path)
    known_info = ['Data file format: tdms', 'Fluorescence: False']
    assert set(info) == set(known_info)
    cleanup()


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
