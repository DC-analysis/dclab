import numpy as np
import pytest

from dclab.features import emodulus


@pytest.mark.filterwarnings('ignore::dclab.features.emodulus.'
                            + 'YoungsModulusLookupTableExceededWarning')
def test_simple_emod_he_3d_fem_22():
    x = np.linspace(0, 250, 100)
    y = np.linspace(0, 0.1, 100)
    x, y = np.meshgrid(x, y)

    emod = emodulus.get_emodulus(area_um=x,
                                 deform=y,
                                 medium="CellCarrier",
                                 channel_width=30,
                                 flow_rate=0.16,
                                 lut_data="HE-3D-FEM-22",
                                 px_um=0,  # without pixelation correction
                                 temperature=23,
                                 visc_model="herold-2017")

    assert np.allclose(emod[10, 50], 1.301024899951213)
    assert np.allclose(emod[50, 50], 0.6713689164185259)
    assert np.isnan(emod[80, 50])

    assert np.allclose(emod[10, 80], 1.7172087155883002)
    assert np.allclose(emod[50, 80], 0.8595551172406175)
    assert np.isnan(emod[80, 80])
