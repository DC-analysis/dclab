from distutils.version import LooseVersion

import numpy as np
import pytest

import dclab
from dclab.lme4 import Rlme4, bootstrapped_median_distributions, rsetup, rlibs


pytest.importorskip("rpy2")


def standard_datasets(set_region=True):
    features = [
        [100, 99, 80, 120, 140, 150, 100, 100, 110, 111,
         140, 145],  # Larger values (Control Channel1)
        [20, 10, 5, 16, 14, 22, 27, 26, 5, 10, 11, 8, 15, 17,
         20, 9],  # Smaller values (Control Reservoir1)
        [115, 110, 90, 110, 145, 155, 110, 120, 115, 120, 120,
         150, 100, 90, 100],  # Larger values (Control Channel2)
        [30, 30, 15, 26, 24, 32, 37, 36, 15, 20, 21, 18, 25,
         27, 30, 19],  # Smaller values (Control Reservoir2)
        [150, 150, 130, 170, 190, 250, 150, 150, 160, 161, 180, 195, 130,
         120, 125, 130, 125],  # Larger values (Treatment Channel1)
        [2, 1, 5, 6, 4, 2, 7, 6, 5, 10, 1, 8, 5, 7, 2, 9, 11,
         8, 13],  # Smaller values (Treatment Reservoir1)
        [155, 155, 135, 175, 195, 255, 155, 155, 165, 165, 185, 200, 135, 125,
         130, 135, 140, 150, 135, 140],  # Larger values (Treatment Channel2)
        [25, 15, 19, 26, 44, 42, 35, 20, 15, 10, 11, 28, 35, 10, 25,
         13]]  # Smaller values (Treatment Reservoir2)
    regions = ["channel", "reservoir"] * 4
    datasets = []
    for ii, ff in enumerate(features):
        ds = dclab.new_dataset({"deform": ff})
        if set_region:
            region = regions[ii]
        else:
            region = "channel"
        ds.config["setup"]["chip region"] = region
        assert ds.config["setup"]["chip region"] == region
        datasets.append(ds)
    return datasets


def test_differential():
    # Larger values (Channel1)
    a = np.array([100, 99, 80, 120, 140, 150, 100, 100, 110, 111, 140, 145])
    # Smaller values (Reservoir1)
    b = np.array([20, 10, 5, 16, 14, 22, 27, 26, 5, 10, 11, 8, 15, 17, 20, 9])
    result = bootstrapped_median_distributions(a, b, bs_iter=1000)
    assert np.allclose([np.median(result[0])], [110.5])
    assert np.allclose([np.median(result[1])], [14.5])


def test_basic_setup():
    assert rsetup.has_r()
    rsetup.install_lme4()
    assert rsetup.has_lme4()


def test_import_rpy2():
    import rpy2
    from rpy2 import situation
    assert LooseVersion(rpy2.__version__) >= LooseVersion(rlibs.R_MIN_VERSION)
    assert situation.get_r_home() is not None


def test_fail_add_same_dataset():
    datasets = standard_datasets(set_region=False)

    rlme4 = Rlme4(model="lmer", feature="deform")
    rlme4.add_dataset(datasets[0], "control", 1)

    with pytest.raises(ValueError, match="has already"):
        rlme4.add_dataset(datasets[1], "control", 1)


def test_fail_too_few_dataset():
    datasets = standard_datasets(set_region=False)

    rlme4 = Rlme4(model="lmer", feature="deform")
    rlme4.add_dataset(datasets[0], "control", 1)
    rlme4.add_dataset(datasets[1], "treatment", 1)

    with pytest.raises(ValueError, match="Linear mixed effects models"
                                         + " require repeated measurements"):
        rlme4.fit()


def test_fail_get_non_existent_data():
    datasets = standard_datasets(set_region=False)

    rlme4 = Rlme4(model="lmer", feature="deform")
    rlme4.add_dataset(datasets[0], "control", 1)
    rlme4.add_dataset(datasets[1], "treatment", 1)
    rlme4.add_dataset(datasets[2], "control", 2)
    rlme4.add_dataset(datasets[3], "treatment", 2)
    rlme4.add_dataset(datasets[3], "treatment", 3)

    with pytest.raises(ValueError, match="Dataset for group 'control'"):
        rlme4.get_feature_data(group="control", repetition=3)


def test_glmer_basic_larger():
    """Original values in a generalized linear mixed model"""
    groups = ['treatment', 'control', 'treatment', 'control',
              'treatment', 'control', 'treatment', 'control']
    repetitions = [1, 1, 2, 2, 3, 3, 4, 4]
    datasets = standard_datasets(set_region=False)

    rlme4 = Rlme4(model="glmer+loglink", feature="deform")
    for ii in range(len(datasets)):
        rlme4.add_dataset(datasets[ii], groups[ii], repetitions[ii])

    res = rlme4.fit()
    assert np.allclose(res["fixed effects intercept"], 15.083832084641697)
    assert np.allclose(res["anova p-value"], 0.00365675950677214)
    assert not rlme4.is_differential()
    assert res["model converged"]


def test_glmer_differential():
    """Differential Deformation in a generalized linear mixed model"""
    groups = ['control', 'control', 'control', 'control', 'treatment',
              'treatment', 'treatment', 'treatment']
    repetitions = [1, 1, 2, 2, 1, 1, 2, 2]
    datasets = standard_datasets()
    rlme4 = Rlme4(model="glmer+loglink", feature="deform")
    for ii in range(len(datasets)):
        rlme4.add_dataset(datasets[ii], groups[ii], repetitions[ii])

    res = rlme4.fit()
    assert np.allclose(res["fixed effects intercept"], 93.55789706339498)
    assert rlme4.is_differential()
    assert rlme4.model == "glmer+loglink"
    assert np.allclose(res["anova p-value"], 0.000556063024310929)
    assert res["model converged"]


def test_lmer_basic():
    groups = ['control', 'treatment', 'control', 'treatment']
    repetitions = [1, 1, 2, 2]
    features = [
        [100, 99, 80, 120, 140, 150, 100, 100, 110, 111, 140, 145],
        [115, 110, 90, 110, 145, 155, 110, 120, 115, 120, 120, 150,
         100, 90, 100],
        [150, 150, 130, 170, 190, 250, 150, 150, 160,
         161, 180, 195, 130, 120, 125, 130, 125],
        [155, 155, 135, 175, 195, 255, 155, 155, 165, 165,
         185, 200, 135, 125, 130, 135, 140, 150, 135, 140]
    ]
    datasets = []
    for ff in features:
        ds = dclab.new_dataset({"deform": ff})
        ds.config["setup"]["chip region"] = "channel"
        assert ds.config["setup"]["chip region"] == "channel"
        datasets.append(ds)

    rlme4 = Rlme4(model="lmer", feature="deform")
    for ii in range(len(datasets)):
        rlme4.add_dataset(datasets[ii], groups[ii], repetitions[ii])

    res = rlme4.fit()
    assert np.allclose(res["anova p-value"], 0.8434625179432129)
    assert np.allclose(res["fixed effects intercept"], 136.6365047987075)
    assert np.allclose(res["fixed effects treatment"], 1.4085644911191584,
                       atol=0, rtol=1e-3)
    assert np.allclose(res["fixed effects intercept"],
                       np.mean(res["fixed effects repetitions"], axis=1)[0])
    assert np.allclose(res["fixed effects treatment"],
                       np.mean(res["fixed effects repetitions"], axis=1)[1])
    assert not res["is differential"]
    assert res["feature"] == "deform"
    assert res["model"] == "lmer"
    assert res["model converged"]


def test_lmer_basic_filtering():
    groups = ['control', 'treatment', 'control', 'treatment']
    repetitions = [1, 1, 2, 2]
    features = [
        [100, 99, 80, 120, 140, 150, 100, 100, 110, 111, 140, 145],
        [115, 110, 90, 110, 145, 155, 110, 120, 115, 120, 120, 150,
         100, 90, 100],
        [150, 150, 130, 170, 190, 250, 150, 150, 160,
         161, 180, 195, 130, 120, 125, 130, 125],
        [155, 155, 135, 175, 195, 255, 155, 155, 165, 165,
         185, 200, 135, 125, 130, 135, 140, 150, 135, 140]
    ]
    datasets = []
    for ff in features:
        ds = dclab.new_dataset({"deform": ff})
        ds.config["setup"]["chip region"] = "channel"
        # apply some filters
        datasets.append(ds)

    rlme4 = Rlme4(model="lmer", feature="deform")
    for ii in range(len(datasets)):
        rlme4.add_dataset(datasets[ii], groups[ii], repetitions[ii])

    res = rlme4.fit()
    assert np.allclose(res["anova p-value"], 0.8434625179432129)

    # filters should have an effect

    for ds in datasets:
        ds.filter.manual[:4] = False
        ds.apply_filter()

    res2 = rlme4.fit()
    assert not np.allclose(res["anova p-value"], res2["anova p-value"])


def test_lmer_basic_larger():
    """Original values in a linear mixed model

    'Reservoir' measurements are now Controls and 'Channel'
    measurements are Treatments. This does not use differential
    deformation.
    """
    groups = ['treatment', 'control', 'treatment', 'control',
              'treatment', 'control', 'treatment', 'control']
    repetitions = [1, 1, 2, 2, 3, 3, 4, 4]
    datasets = standard_datasets(set_region=False)

    rlme4 = Rlme4(model="lmer", feature="deform")
    for ii in range(len(datasets)):
        rlme4.add_dataset(datasets[ii], groups[ii], repetitions[ii])

    res = rlme4.fit()
    assert np.allclose(res["fixed effects intercept"], 17.171341507432501)
    assert np.allclose(res["anova p-value"], 0.000331343267412872)
    assert not rlme4.is_differential()
    assert res["model converged"]


def test_lmer_basic_nan():
    groups = ['control', 'treatment', 'control', 'treatment']
    repetitions = [1, 1, 2, 2]
    features = [
        [100, np.inf, 80, np.nan, 140, 150, 100, 100, 110, 111, 140, 145],
        [115, 110, 90, 110, 145, 155, 110, 120, 115, 120, 120, 150,
         100, 90, 100],
        [150, 150, 130, 170, 190, 250, 150, 150, 160,
         161, 180, 195, 130, 120, 125, 130, 125],
        [155, 155, 135, 175, 195, 255, 155, 155, 165, 165,
         185, 200, 135, 125, 130, 135, 140, 150, 135, 140]
    ]
    datasets = []
    for ff in features:
        ds = dclab.new_dataset({"deform": ff})
        ds.config["setup"]["chip region"] = "channel"
        assert ds.config["setup"]["chip region"] == "channel"
        datasets.append(ds)

    rlme4 = Rlme4(model="lmer", feature="deform")
    for ii in range(len(datasets)):
        rlme4.add_dataset(datasets[ii], groups[ii], repetitions[ii])

    res = rlme4.fit()
    assert np.allclose(res["fixed effects intercept"], 137.37179302516199186)
    # assert res["model converged"]  # does not converge on Travis-CI


def test_lmer_differential():
    """Differential Deformation in a linear mixed model"""
    groups = ['control', 'control', 'control', 'control', 'treatment',
              'treatment', 'treatment', 'treatment']
    repetitions = [1, 1, 2, 2, 1, 1, 2, 2]
    datasets = standard_datasets()
    rlme4 = Rlme4(model="lmer", feature="deform")
    for ii in range(len(datasets)):
        rlme4.add_dataset(datasets[ii], groups[ii], repetitions[ii])

    res = rlme4.fit()
    assert np.allclose(res["fixed effects intercept"], 93.693750004463098)
    assert rlme4.is_differential()
    assert rlme4.model == "lmer"
    assert np.allclose(res["anova p-value"], 0.000602622503360039)
    assert np.allclose(res["fixed effects intercept"],
                       np.mean(res["fixed effects repetitions"], axis=1)[0])
    assert np.allclose(res["fixed effects treatment"],
                       np.mean(res["fixed effects repetitions"], axis=1)[1])
    assert res["model converged"]


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
