import numpy as np
import pytest

from dclab.features import emodulus


@pytest.mark.filterwarnings('ignore::dclab.features.emodulus.'
                            + 'YoungsModulusLookupTableExceededWarning')
def test_simple_emod_he_2d_fem_22():
    x = np.linspace(0, 250, 100)
    y = np.linspace(0, 0.1, 100)
    x, y = np.meshgrid(x, y)

    emod = emodulus.get_emodulus(area_um=x,
                                 deform=y,
                                 medium="CellCarrier",
                                 channel_width=30,
                                 flow_rate=0.16,
                                 lut_data="HE-2D-FEM-22",
                                 px_um=0,  # without pixelation correction
                                 temperature=23,
                                 visc_model="herold-2017")

    assert np.allclose(emod[10, 50], 1.2520069112713232)
    assert np.allclose(emod[50, 50], 0.6442165714687615)
    assert np.allclose(emod[80, 50], 0.575373042660183)

    assert np.allclose(emod[10, 80], 1.638297273096505)
    assert np.allclose(emod[50, 80], 0.8191788744746755)
    assert np.allclose(emod[80, 80], 0.716695164528623)
