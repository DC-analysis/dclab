from unittest import mock

import pytest

import dclab
from dclab import new_dataset, RTDCWriter
import numpy as np

from helper_methods import example_data_dict, retrieve_data

av = pytest.importorskip("av")


def test_avi_export_check(tmp_path):
    ds = new_dataset(retrieve_data("fmt-hdf5_wide-channel_2023.zip"))
    avi_path = tmp_path / "exported.avi"
    ds.export.avi(path=avi_path)
    num_frames = 0
    with av.open(avi_path) as container:
        for ii, frame in enumerate(container.decode(video=0)):
            num_frames += 1
            array = frame.to_ndarray(format="rgb24")
            for jj in range(3):
                assert np.allclose(ds["image"][ii],
                                   array[:, :, jj],
                                   atol=5,  # one reason why we use HDF5
                                   rtol=0)
    assert num_frames == len(ds)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_avi_export_progress(tmp_path):
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    # create an .rtdc file with more than 10_000 events
    path_large = path.with_name("large.rtdc")
    path_avi = path.with_name("export.avi")

    with RTDCWriter(path_large) as hw, new_dataset(path) as ds:
        hw.store_metadata(ds.config.as_dict(pop_filtering=True))
        size = len(ds)
        num_iters = 25_000 // size
        for feat in ["time", "image", "deform"]:
            feat_data = np.concatenate([ds[feat][:]] * num_iters)
            hw.store_feature(feat, feat_data)

    with dclab.new_dataset(path_large) as dse:
        assert len(dse) > 20_000, "we need a large dataset"

        callback = mock.MagicMock()
        dse.export.avi(path=path_avi, progress_callback=callback)

    callback.assert_any_call(0.0, "exporting video")
    callback.assert_any_call(0.4, "exporting video")
    callback.assert_any_call(0.8, "exporting video")
    callback.assert_any_call(1.0, "video export complete")


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.exc.CorruptFrameWarning')
def test_avi_export_tdms(tmp_path):
    pytest.importorskip("nptdms")
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))

    f1 = tmp_path / "test.avi"
    ds.export.avi(path=f1)
    assert f1.stat()[6] > 1e4, "Resulting file to small, Something went wrong!"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.exc.CorruptFrameWarning')
def test_avi_override_tdms(tmp_path):
    pytest.importorskip("nptdms")
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))

    f1 = tmp_path / "test.avi"
    ds.export.avi(f1, override=True)
    try:
        ds.export.avi(f1.with_name(f1.stem), override=False)
    except OSError:
        pass
    else:
        raise ValueError("Should append .avi and not override!")


def test_avi_no_images(tmp_path):
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=127, keys=keys)
    ds = dclab.new_dataset(ddict)

    try:
        ds.export.avi(tmp_path / "test.avi")
    except OSError:
        pass
    else:
        raise ValueError("There should be no image data to write!")
