"""Test machine learning tools"""
import pathlib
import tempfile

import numpy as np
import pytest

import dclab
from dclab import new_dataset
from dclab.rtdc_dataset import feat_anc_ml

from helper_methods import example_data_dict


@pytest.fixture(autouse=True)
def cleanup_plugin_features():
    """Fixture used to cleanup plugin feature tests"""
    # code run before the test
    pass
    # then the test is run
    yield
    # code run after the test
    # remove our test plugin examples
    feat_anc_ml.remove_all_ml_features()


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


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.feat_anc_ml.modc.'
                            + 'ModelFormatExportFailedWarning')
def test_modc_export_model_bad_model():
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="dclab_ml"))
    with pytest.raises(ValueError,
                       match="Export failed for all model formats!"):
        feat_anc_ml.modc.export_model(path=tmpdir,
                                      model=object()
                                      )
