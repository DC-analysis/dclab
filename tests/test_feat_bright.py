import numpy as np
import pytest

import dclab
from dclab import new_dataset
from dclab.features.bright import get_bright

from helper_methods import retrieve_data


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.event_image.'
                            + 'InitialFrameMissingWarning')
@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'ancillaries.ancillary_feature.'
                            + 'BadFeatureSizeWarning')
def test_af_brightness():
    # Brightness of the image
    ds = dclab.new_dataset(retrieve_data("rtdc_data_traces_video_bright.zip"))
    # This is something low-level and should not be done in a script.
    # Remove the brightness columns from RTDCBase to force computation with
    # the image and contour columns.
    real_avg = ds._events.pop("bright_avg")
    real_sd = ds._events.pop("bright_sd")
    # This will cause a zero-padding warning:
    comp_avg = ds["bright_avg"]
    comp_sd = ds["bright_sd"]
    idcompare = ~np.isnan(comp_avg)
    # ignore first event (no image data)
    idcompare[0] = False
    assert np.allclose(real_avg[idcompare], comp_avg[idcompare])
    assert np.allclose(real_sd[idcompare], comp_sd[idcompare])


def test_simple_bright():
    ds = new_dataset(retrieve_data("rtdc_data_traces_video_bright.zip"))
    for ii in range(2, 7):
        # This stripped dataset has only 7 video frames / contours
        image = ds["image"][ii]
        mask = ds["mask"][ii]
        avg, std = get_bright(mask=mask, image=image, ret_data="avg,sd")
        assert np.allclose(avg, ds["bright_avg"][ii])
        assert np.allclose(std, ds["bright_sd"][ii])


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
