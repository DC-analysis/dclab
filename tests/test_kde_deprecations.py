import pytest


def test_kde_contours_deprecated_warning():
    with pytest.deprecated_call():
        import dclab.kde_contours  # noqa: F401


def test_kde_methods_deprecated_warning():
    with pytest.deprecated_call():
        import dclab.kde_methods  # noqa: F401
