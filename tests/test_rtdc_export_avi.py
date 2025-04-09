import os
from os.path import join
import tempfile

import pytest

import dclab
from dclab import new_dataset
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
@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.exc.CorruptFrameWarning')
def test_avi_export_tdms():
    pytest.importorskip("nptdms")
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))
    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.avi")
    ds.export.avi(path=f1)
    assert os.stat(
        f1)[6] > 1e4, "Resulting file to small, Something went wrong!"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.exc.CorruptFrameWarning')
def test_avi_override_tdms():
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
