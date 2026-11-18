import numpy as np
import pytest

import dclab
from dclab.definitions import get_feature_label


@pytest.mark.parametrize("name,label",
                         [
                             ["ml_score_rbc", "ML score RBC"],
                             ["deform", "Deformation"],
                             ["area_um", "Area [µm²]"],
                         ])
def test_get_feature_label_basic(name, label):
    assert get_feature_label(name) == label


def test_get_feature_label_temporary():
    name = "unknown"
    label = "User-defined feature unknown"
    dclab.register_temporary_feature(name)
    assert get_feature_label(name) == label


def test_get_feature_label_invalid():
    name = "unknown2"
    with pytest.raises(ValueError, match="unknown2"):
        get_feature_label(name)


def test_deprecated_objects():
    names = dclab.definitions.feature_names
    with pytest.warns(DeprecationWarning, match="feature_names"):
        assert len(names) == len(dclab.definitions.feat_const.feature_names)
    with pytest.warns(DeprecationWarning, match="feature_names"):
        assert names[names.index("deform")] == "deform"
    descr = dclab.definitions.config_descr
    with pytest.warns(DeprecationWarning, match="config_descr"):
        assert descr.get("experiment")["sample"]


def test_feature_deregister_same_label():
    """Labels are not unique, the correct label must be removed"""
    feat_const = dclab.definitions.feat_const
    feat_logic = dclab.definitions.feat_logic
    feat_logic.feature_register("userdereg1", label="Same label")
    feat_logic.feature_register("userdereg2", label="Other label")
    feat_logic.feature_register("userdereg3", label="Same label")
    try:
        feat_logic.feature_deregister("userdereg3")
        idx = feat_const.feature_names.index("userdereg2")
        assert feat_const.feature_labels[idx] == "Other label"
        assert (len(feat_const.feature_names)
                == len(feat_const.feature_labels))
    finally:
        feat_logic.feature_deregister("userdereg1")
        feat_logic.feature_deregister("userdereg2")
    assert "userdereg2" not in feat_const.feature_names
    assert "Other label" not in feat_const.feature_labels


def test_deprecated_objects_iter_contains():
    names = dclab.definitions.feature_names
    with pytest.warns(DeprecationWarning, match="feature_names"):
        assert "deform" in names
    with pytest.warns(DeprecationWarning, match="feature_names"):
        assert "deform" in list(names)


@pytest.mark.parametrize("name,shape", [
    ("deform", (10,)),
    ("image", (10, 80, 250)),
    ("ml_score_abc", (10,)),
])
def test_check_feature_shape(name, shape):
    dclab.definitions.check_feature_shape(name, np.zeros(shape))


def test_check_feature_shape_errors():
    with pytest.raises(ValueError, match="is not a scalar feature"):
        dclab.definitions.check_feature_shape("image", np.zeros(10))
    with pytest.raises(ValueError, match="is a scalar feature"):
        dclab.definitions.check_feature_shape("deform", np.zeros((10, 2)))


@pytest.mark.parametrize("name,exists,scalar", [
    ("deform", True, True),
    ("image", True, False),
    ("ml_score_abc", True, True),
    ("ml_score_ab", False, False),
    ("peter", False, False),
])
def test_feature_exists(name, exists, scalar):
    assert dclab.definitions.feature_exists(name) == exists
    assert dclab.definitions.scalar_feature_exists(name) == scalar


def test_feature_register_errors():
    feat_logic = dclab.definitions.feat_logic
    with pytest.raises(ValueError, match="only contain lower-case"):
        feat_logic.feature_register("User_Feature")
    with pytest.raises(ValueError, match="already exists"):
        feat_logic.feature_register("deform")
