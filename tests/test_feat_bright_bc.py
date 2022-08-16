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
    assert "bright_bc_avg" not in ds.features_innate
    assert "bright_bc_sd" not in ds.features_innate
    # ignore first event (no image data)
    for ii in range(1, len(ds)):
        image_corr = np.array(ds["image"][ii], dtype=int) - ds["image_bg"][ii]
        mask = ds["mask"][ii]
        assert np.max(np.abs(image_corr)) < 50, "sanity check"
        assert ds["bright_bc_avg"][ii] == np.mean(image_corr[mask])
        assert ds["bright_bc_sd"][ii] == np.std(image_corr[mask])
