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
