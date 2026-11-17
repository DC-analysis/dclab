import h5py
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
def test_af_brightness_bc_with_offset():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    with h5py.File(path, "a") as h5:
        bg_off = np.linspace(0, 1, len(h5["events/deform"]))
        h5["events/bg_off"] = bg_off

    ds = dclab.new_dataset(path)
    # sanity checks
    assert "bright_perc_10" not in ds.features_innate
    assert "bright_perc_90" not in ds.features_innate
    # ignore first event (no image data)
    for ii in range(1, len(ds)):
        image_bg = ds["image_bg"][ii] + bg_off[ii]
        image_corr = np.array(ds["image"][ii], dtype=int) - image_bg
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

        p10_calc, p90_calc = dclab.features.bright_perc.get_bright_perc(
            ds["mask"][ii],
            ds["image"][ii],
            ds["image_bg"][ii],
        )
        assert (p10_calc, p90_calc) == (p10, p90)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_af_brightness_bc_list():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    ds = dclab.new_dataset(path)
    # sanity checks
    assert "bright_perc_10" not in ds.features_innate
    assert "bright_perc_90" not in ds.features_innate
    # ignore first event (no image data)
    mask = list(ds["mask"][1:6])
    image = list(ds["image"][1:6])
    image_bg = list(ds["image_bg"][1:6])

    p10_calc, p90_calc = dclab.features.bright_perc.get_bright_perc(
        mask, image, image_bg,
    )
    assert isinstance(p10_calc, np.ndarray)
    assert isinstance(p90_calc, np.ndarray)
    assert p10_calc.shape == (4,)
    assert p90_calc.shape == (4,)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_af_brightness_bc_single_with_offset():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    with h5py.File(path, "a") as h5:
        bg_off = np.linspace(0, 1, len(h5["events/deform"]))
        h5["events/bg_off"] = bg_off

    ds = dclab.new_dataset(path)
    # sanity checks
    assert "bright_perc_10" not in ds.features_innate
    assert "bright_perc_90" not in ds.features_innate
    # ignore first event (no image data)
    for ii in range(1, len(ds)):
        image_bg = ds["image_bg"][ii] + bg_off[ii]
        image_corr = np.array(ds["image"][ii], dtype=int) - image_bg
        mask = ds["mask"][ii]
        p10 = np.percentile(image_corr[mask], 10)
        p90 = np.percentile(image_corr[mask], 90)
        assert np.max(np.abs(image_corr)) < 50, "sanity check"
        assert dclab.features.bright_perc.get_bright_perc(
            ds["mask"][ii],
            ds["image"][ii],
            ds["image_bg"][ii],
            ds["bg_off"][ii],
        ) == (p10, p90)


def test_af_brightness_bc_empty_mask():
    """Empty masks yield nan (like `bright_avg`)"""
    image = np.full((10, 10), 100, dtype=np.uint8)
    image_bg = np.full((10, 10), 90, dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)
    p10, p90 = dclab.features.bright_perc.get_bright_perc(
        mask, image, image_bg)
    assert np.isnan(p10)
    assert np.isnan(p90)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_af_brightness_bc_with_offset_array():
    """`bg_off` given as a numpy array (e.g. for dictionary datasets)"""
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    with dclab.new_dataset(path) as ds0:
        # ignore first event (no image data)
        data = {feat: ds0[feat][1:]
                for feat in ["image", "image_bg", "mask", "deform"]}
    bg_off = np.linspace(0, 1, len(data["deform"]))
    data["bg_off"] = bg_off

    p10_calc, p90_calc = dclab.features.bright_perc.get_bright_perc(
        list(data["mask"]), list(data["image"]), list(data["image_bg"]),
        bg_off=bg_off,
    )

    ds = dclab.new_dataset(data)
    assert isinstance(ds["bg_off"], np.ndarray), "sanity check"
    for ii in range(len(ds)):
        image_bg = ds["image_bg"][ii] + bg_off[ii]
        image_corr = np.array(ds["image"][ii], dtype=int) - image_bg
        mask = ds["mask"][ii]
        p10 = np.percentile(image_corr[mask], 10)
        p90 = np.percentile(image_corr[mask], 90)
        # (bg_off is subtracted after computing the percentiles)
        assert np.allclose(ds["bright_perc_10"][ii], p10, rtol=0, atol=1e-12)
        assert np.allclose(ds["bright_perc_90"][ii], p90, rtol=0, atol=1e-12)
        assert np.allclose(p10_calc[ii], p10, rtol=0, atol=1e-12)
        assert np.allclose(p90_calc[ii], p90, rtol=0, atol=1e-12)
