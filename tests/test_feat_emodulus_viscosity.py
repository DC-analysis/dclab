import numpy as np
import warnings

import pytest

from dclab.features import emodulus


def test_buyukurganci_cell_carrier():
    """Test values inspired from the table in Herold's paper"""
    ch_sizes = [15, 15, 15, 20, 20, 20, 20, 20, 30, 30, 30, 40, 40, 40]
    fl_rates = [0.016, 0.032, 0.048, 0.02, 0.04, 0.06,
                0.08, 0.12, 0.16, 0.24, 0.32, 0.32, 0.40, 0.60]
    temps = [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24]
    eta_a = [
        5.4001970145102955,
        4.257647591503594,
        3.704914977001318,
        6.725414611510781,
        5.3024816031010324,
        4.6141074582608255,
        4.180606367836839,
        3.63787533192147,
        5.0023466326283845,
        4.352936348316222,
        3.94397260600829,
        5.3024816031010324,
        4.9118302388920805,
        4.274170894950141,
    ]
    eta_b = [
        6.384624807560792,
        4.799374333891269,
        4.061421656375954,
        8.309306638844017,
        6.246173301784982,
        5.285760549704508,
        4.695299212276897,
        3.9733491445425506,
        5.824144616254946,
        4.928623072237973,
        4.378056820977706,
        6.246173301784982,
        5.697847203911832,
        4.821745173172601,
    ]

    for c, f, t, a in zip(ch_sizes, fl_rates, temps, eta_a):
        eta = emodulus.viscosity.get_viscosity(medium="0.49% MC-PBS",
                                               channel_width=c,
                                               flow_rate=f,
                                               temperature=t,
                                               model='buyukurganci-2022')
        assert np.allclose(eta, a)

    for c, f, t, b in zip(ch_sizes, fl_rates, temps, eta_b):
        eta = emodulus.viscosity.get_viscosity(medium="0.59% MC-PBS",
                                               channel_width=c,
                                               flow_rate=f,
                                               temperature=t,
                                               model='buyukurganci-2022')
        assert np.allclose(eta, b)


@pytest.mark.parametrize("medium", ["0.49% MC-PBS", "0.59% MC-PBS"])
def test_buyukurganci_mcpbs_range(medium):
    # test values
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        emodulus.viscosity.get_viscosity(
            medium=medium, temperature=21, model="buyukurganci-2022")
        assert issubclass(w[-1].category,
                          emodulus.viscosity.TemperatureOutOfRangeWarning)

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        emodulus.viscosity.get_viscosity(
            medium=medium, temperature=38, model="buyukurganci-2022")
        assert issubclass(w[-1].category,
                          emodulus.viscosity.TemperatureOutOfRangeWarning)

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        emodulus.viscosity.get_viscosity(medium=medium,
                                         temperature=np.linspace(1, 10, 8),
                                         model="buyukurganci-2022")
        assert issubclass(w[-1].category,
                          emodulus.viscosity.TemperatureOutOfRangeWarning)


def test_herold_cell_carrier():
    """Test using table from Christophs script"""
    ch_sizes = [15, 15, 15, 20, 20, 20, 20, 20, 30, 30, 30, 40, 40, 40]
    fl_rates = [0.016, 0.032, 0.048, 0.02, 0.04, 0.06,
                0.08, 0.12, 0.16, 0.24, 0.32, 0.32, 0.40, 0.60]
    temps = [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24]
    eta_a = [5.8, 4.6, 4.1, 7.1, 5.7, 5.0,
             4.5, 4.0, 5.4, 4.7, 4.3, 5.7, 5.3, 4.6]
    eta_b = [7.5, 5.8, 5.0, 9.4, 7.3, 6.3,
             5.7, 4.9, 6.9, 5.9, 5.3, 7.3, 6.7, 5.8]

    for c, f, t, a in zip(ch_sizes, fl_rates, temps, eta_a):
        eta = emodulus.viscosity.get_viscosity(medium="CellCarrier",
                                               channel_width=c,
                                               flow_rate=f,
                                               temperature=t,
                                               model="herold-2017")
        assert np.allclose(np.round(eta, 1), a)

    for c, f, t, b in zip(ch_sizes, fl_rates, temps, eta_b):
        eta = emodulus.viscosity.get_viscosity(medium="CellCarrier B",
                                               channel_width=c,
                                               flow_rate=f,
                                               temperature=t,
                                               model="herold-2017")
        assert np.allclose(np.round(eta, 1), b)


@pytest.mark.parametrize("medium", ["0.49% MC-PBS", "0.59% MC-PBS"])
def test_herold_cellcarrier_range(medium):
    # test values
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        emodulus.viscosity.get_viscosity(
            medium=medium, temperature=15, model="herold-2017")
        assert issubclass(w[-1].category,
                          emodulus.viscosity.TemperatureOutOfRangeWarning)

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        emodulus.viscosity.get_viscosity(
            medium=medium, temperature=28, model="herold-2017")
        assert issubclass(w[-1].category,
                          emodulus.viscosity.TemperatureOutOfRangeWarning)

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        emodulus.viscosity.get_viscosity(medium=medium,
                                         temperature=np.linspace(1, 10, 8),
                                         model="herold-2017")
        assert issubclass(w[-1].category,
                          emodulus.viscosity.TemperatureOutOfRangeWarning)


def test_kestin_water():
    """Test with data from Kestin et al., J. Phys. Chem. 7(3) 1978"""
    ref = np.array([[0, 1791.5],
                    [5, 1519.3],
                    [10, 1307.0],
                    [15, 1138.3],
                    [20, 1002.0],
                    [25, 890.2],
                    [30, 797.3],
                    [35, 719.1],
                    [40, 652.7],
                    ])
    ref[:, 1] *= 1e-3  # uPas to mPas

    res = emodulus.viscosity.get_viscosity(medium="water",
                                           temperature=ref[:, 0])
    assert np.allclose(res, ref[:, 1], rtol=8e-5, atol=0)


def test_kestin_water_range():
    # test values
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        emodulus.viscosity.get_viscosity(medium="water", temperature=-1)
        assert issubclass(w[-1].category,
                          emodulus.viscosity.TemperatureOutOfRangeWarning)

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        emodulus.viscosity.get_viscosity(medium="water", temperature=41)
        assert issubclass(w[-1].category,
                          emodulus.viscosity.TemperatureOutOfRangeWarning)

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        emodulus.viscosity.get_viscosity(medium="water",
                                         temperature=np.arange(-2, 10))
        assert issubclass(w[-1].category,
                          emodulus.viscosity.TemperatureOutOfRangeWarning)
