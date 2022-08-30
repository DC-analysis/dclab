import numpy as np
import pytest

import dclab

from helper_methods import retrieve_data


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_af_brightness_bc():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    ds = dclab.new_dataset(path)
    # sanity checks
    assert "bright_perc_10" not in ds.features_innate
    assert "bright_perc_90" not in ds.features_innate
    # ignore first event (no image data)
    for ii in range(1, len(ds)):
        image_corr = np.array(ds["image"][ii], dtype=int) - ds["image_bg"][ii]
        mask = ds["mask"][ii]
        p10 = np.percentile(image_corr[mask], 10)
        p90 = np.percentile(image_corr[mask], 90)
        assert np.max(np.abs(image_corr)) < 50, "sanity check"
        assert ds["bright_perc_10"][ii] == p10
        assert ds["bright_perc_90"][ii] == p90


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_af_brightness_bc_single():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    ds = dclab.new_dataset(path)
    # sanity checks
    assert "bright_perc_10" not in ds.features_innate
    assert "bright_perc_90" not in ds.features_innate
    # ignore first event (no image data)
    for ii in range(1, len(ds)):
        image_corr = np.array(ds["image"][ii], dtype=int) - ds["image_bg"][ii]
        mask = ds["mask"][ii]
        p10 = np.percentile(image_corr[mask], 10)
        p90 = np.percentile(image_corr[mask], 90)
        assert np.max(np.abs(image_corr)) < 50, "sanity check"
        assert dclab.features.bright_perc.get_bright_perc(
            ds["mask"][ii],
            ds["image"][ii],
            ds["image_bg"][ii],
        ) == (p10, p90)
