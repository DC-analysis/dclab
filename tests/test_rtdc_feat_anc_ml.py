"""Test machine learning tools"""
import pathlib

import numpy as np
import pytest

import dclab
from dclab import new_dataset

from helper_methods import example_data_dict


data_path = pathlib.Path(__file__).parent / "data"


def make_data(add_feats=None, sizes=None):
    if sizes is None:
        sizes = [100, 130]
    if add_feats is None:
        add_feats = ["area_um", "deform"]
    keys = add_feats + ["time", "frame", "fl3_width"]
    data = []
    for size in sizes:
        data.append(new_dataset(example_data_dict(size=size, keys=keys)))
    return data


def test_af_ml_class_basic():
    data = {"ml_score_001": [.1, .3, .1, 0.01, .59],
            "ml_score_002": [.2, .1, .4, 0, .8],
            }
    ds = dclab.new_dataset(data)
    assert "ml_class" in ds
    assert np.allclose(ds["ml_class"], [1, 0, 1, 0, 1])
    assert issubclass(ds["ml_class"].dtype.type, np.integer)


def test_af_ml_class_bad_feature():
    data = {"ml_score_0-1": [.1, .3, .1, 0.01, .59],
            }
    with pytest.raises(ValueError,
                       match="Invalid feature name 'ml_score_0-1'"):
        dclab.new_dataset(data)


def test_af_ml_class_bad_score_max():
    data = {"ml_score_001": [.1, .3, 99, 0.01, .59],
            "ml_score_002": [.2, .1, .4, 0, .8],
            }
    ds = dclab.new_dataset(data)
    assert "ml_class" in ds
    with pytest.raises(ValueError, match="> 1"):
        ds["ml_class"]


def test_af_ml_class_bad_score_min():
    data = {"ml_score_001": [.1, .3, -.1, 0.01, .59],
            "ml_score_002": [.2, .1, .4, 0, .8],
            }
    ds = dclab.new_dataset(data)
    assert "ml_class" in ds
    with pytest.raises(ValueError, match="< 0"):
        ds["ml_class"]


def test_af_ml_class_bad_score_nan():
    data = {"ml_score_001": [.1, .3, np.nan, np.nan, np.nan],
            "ml_score_002": [.2, .1, .4, np.nan, 0],
            }
    ds = dclab.new_dataset(data)
    assert "ml_class" in ds
    assert np.allclose(ds["ml_class"],
                       [1, 0, 1, -1, -1],
                       equal_nan=True,
                       atol=0,
                       rtol=0
                       )


def test_af_ml_class_changed_features():
    data = {"ml_score_011": [.1, .3, .1, 0.01, .59],
            "ml_score_012": [.2, .1, .4, 0, .8],
            }
    ds = dclab.new_dataset(data)
    assert "ml_class" in ds
    assert np.all(ds["ml_class"] == [1, 0, 1, 0, 1])
    # This triggers a recomputation of the ml_class feature the
    # next time it is accessed:
    ds._events["ml_score_003"] = np.array([1, 1, 1, 1, 0])
    assert np.all(ds["ml_class"] == [0, 0, 0, 0, 2])


def test_af_ml_class_has_ml_score_false():
    data = {"deform": [.1, .3, .1, 0.01, .59],
            "area_um": [20, 10, 40, 100, 80],
            }
    ds = dclab.new_dataset(data)
    assert "ml_class" not in ds


def test_af_ml_class_single():
    data = {"ml_score_001": [.1, .3, .1, 0.01, .59],
            }
    ds = dclab.new_dataset(data)
    assert "ml_class" in ds
    assert np.allclose(ds["ml_class"], 0)


def test_af_ml_score_label_fallback():
    """Test whether the correct label is returned"""
    label1 = dclab.dfn.get_feature_label("ml_score_low")
    label2 = dclab.dfn.get_feature_label("ml_score_hig")
    assert label1 == "ML score LOW"
    assert label2 == "ML score HIG"
