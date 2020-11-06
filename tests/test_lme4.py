import numpy as np

import dclab
from dclab.lme4 import Rlme4, rsetup


def test_basic_setup():
    assert rsetup.has_r()
    rsetup.install_lme4()
    assert rsetup.has_lme4()


def test_lmm_basic():
    groups = ['control', 'treatment', 'control', 'treatment']
    repetitions = [1, 1, 2, 2]
    xs = [
        [100, 99, 80, 120, 140, 150, 100, 100, 110, 111, 140, 145],
        [115, 110, 90, 110, 145, 155, 110, 120, 115, 120, 120, 150,
         100, 90, 100],
        [150, 150, 130, 170, 190, 250, 150, 150, 160,
         161, 180, 195, 130, 120, 125, 130, 125],
        [155, 155, 135, 175, 195, 255, 155, 155, 165, 165,
         185, 200, 135, 125, 130, 135, 140, 150, 135, 140]
    ]
    datasets = []
    for x in xs:
        ds = dclab.new_dataset({"deform": x})
        ds.config["setup"]["chip region"] = "channel"
        assert ds.config["setup"]["chip region"] == "channel"
        datasets.append(ds)

    rlme4 = Rlme4(model="lmer", feature="deform")
    for ii in range(len(datasets)):
        rlme4.add_dataset(datasets[ii], groups[ii], repetitions[ii])

    res = rlme4.fit()
    assert np.allclose(res["fixed effects intercept"], 136.6365047987075)


def test_lmm_basic_larger():
    groups = ['treatment', 'control', 'treatment', 'control',
              'treatment', 'control', 'treatment', 'control']
    repetitions = [1, 1, 2, 2, 3, 3, 4, 4]
    xs = [
        [100, 99, 80, 120, 140, 150, 100, 100, 110, 111,
         140, 145],
        [20, 10, 5, 16, 14, 22, 27, 26, 5, 10, 11, 8, 15, 17,
         20, 9],
        [115, 110, 90, 110, 145, 155, 110, 120, 115, 120, 120,
         150, 100, 90, 100],
        [30, 30, 15, 26, 24, 32, 37, 36, 15, 20, 21, 18, 25,
         27, 30, 19],
        [150, 150, 130, 170, 190, 250, 150, 150, 160, 161, 180, 195, 130,
         120, 125, 130, 125],
        [2, 1, 5, 6, 4, 2, 7, 6, 5, 10, 1, 8, 5, 7, 2, 9, 11,
         8, 13],
        [155, 155, 135, 175, 195, 255, 155, 155, 165, 165, 185, 200, 135, 125,
         130, 135, 140, 150, 135, 140],  # Larger values (Treatment Channel2)
        [25, 15, 19, 26, 44, 42, 35, 20, 15, 10, 11, 28, 35, 10, 25,
         13]]

    datasets = []
    for x in xs:
        ds = dclab.new_dataset({"deform": x})
        ds.config["setup"]["chip region"] = "channel"
        assert ds.config["setup"]["chip region"] == "channel"
        datasets.append(ds)

    rlme4 = Rlme4(model="lmer", feature="deform")
    for ii in range(len(datasets)):
        rlme4.add_dataset(datasets[ii], groups[ii], repetitions[ii])

    res = rlme4.fit()
    assert np.allclose(res["anova p-value"], 0.000331343267412872)


def test_lmm_basic_nan():
    groups = ['control', 'treatment', 'control', 'treatment']
    repetitions = [1, 1, 2, 2]
    xs = [
        [100, np.inf, 80, np.nan, 140, 150, 100, 100, 110, 111, 140, 145],
        [115, 110, 90, 110, 145, 155, 110, 120, 115, 120, 120, 150,
         100, 90, 100],
        [150, 150, 130, 170, 190, 250, 150, 150, 160,
         161, 180, 195, 130, 120, 125, 130, 125],
        [155, 155, 135, 175, 195, 255, 155, 155, 165, 165,
         185, 200, 135, 125, 130, 135, 140, 150, 135, 140]
    ]
    datasets = []
    for x in xs:
        ds = dclab.new_dataset({"deform": x})
        ds.config["setup"]["chip region"] = "channel"
        assert ds.config["setup"]["chip region"] == "channel"
        datasets.append(ds)

    rlme4 = Rlme4(model="lmer", feature="deform")
    for ii in range(len(datasets)):
        rlme4.add_dataset(datasets[ii], groups[ii], repetitions[ii])

    res = rlme4.fit()
    assert np.allclose(res["fixed effects intercept"], 137.37179302516199186)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
