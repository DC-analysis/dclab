import numpy as np

import pytest


from dclab.definitions import meta_logic, meta_parse


def assert_is_bool_false(value):
    assert isinstance(value, bool)
    assert not value


def assert_is_bool_true(value):
    assert isinstance(value, bool)
    assert value


def test_meta_logic_config_key_exists():
    assert meta_logic.config_key_exists("user", "test")
    assert not meta_logic.config_key_exists("user", 1)

    assert meta_logic.config_key_exists("setup", "channel width")
    assert not meta_logic.config_key_exists("setup", "funnel width")

    assert meta_logic.config_key_exists("online_filter",
                                        "area_um,deform soft limit")
    assert not meta_logic.config_key_exists("online_filter",
                                            "peter,deform soft limit")

    assert meta_logic.config_key_exists("online_filter",
                                        "area_um,deform polygon points")
    assert not meta_logic.config_key_exists("online_filter",
                                            "peter,deform polygon points")


def test_meta_logic_get_config_value_descr():
    assert meta_logic.get_config_value_descr("user", "key") == "key"

    assert meta_logic.get_config_value_descr("experiment", "date") \
        == "Date of measurement ('YYYY-MM-DD')"

    assert meta_logic.get_config_value_descr(
        "online_filter", "area_um,deform soft limit") \
        == "Soft limit, polygon (Area, Deformation)"

    assert meta_logic.get_config_value_descr(
        "online_filter", "area_um,deform polygon points") \
        == "Polygon (Area, Deformation)"


def test_meta_logic_get_config_value_func():
    lamb = meta_logic.get_config_value_func("user", "key")
    assert lamb("peter") == "peter"

    assert meta_logic.get_config_value_func("experiment", "date") is str

    assert meta_logic.get_config_value_func(
        "online_filter", "area_um,deform soft limit") \
        is meta_parse.fbool

    assert meta_logic.get_config_value_func(
        "online_filter", "area_um,deform polygon points") \
        is meta_parse.f2dfloatarray


def test_meta_logic_get_config_value_type():
    assert meta_logic.get_config_value_type("user", "key") is None

    assert meta_logic.get_config_value_type("experiment", "date") is str

    tsl = meta_logic.get_config_value_type(
        "online_filter", "area_um,deform soft limit")
    assert bool in tsl

    assert meta_logic.get_config_value_type(
        "online_filter", "area_um,deform polygon points") \
        is np.ndarray


def test_meta_parse_f2dfloatarray():
    assert meta_parse.f2dfloatarray([[1], [2]]).shape == (2, 1)


def test_meta_parse_fbool():
    assert_is_bool_true(meta_parse.fbool(True))
    assert_is_bool_true(meta_parse.fbool("true"))
    assert_is_bool_true(meta_parse.fbool("True"))
    assert_is_bool_true(meta_parse.fbool("1"))
    assert_is_bool_true(meta_parse.fbool(1))
    assert_is_bool_true(meta_parse.fbool(2.2))

    assert_is_bool_false(meta_parse.fbool(False))
    assert_is_bool_false(meta_parse.fbool("false"))
    assert_is_bool_false(meta_parse.fbool("False"))
    assert_is_bool_false(meta_parse.fbool("0"))
    assert_is_bool_false(meta_parse.fbool("0.0"))
    assert_is_bool_false(meta_parse.fbool(0))
    assert_is_bool_false(meta_parse.fbool(0.))

    with pytest.raises(ValueError, match="Empty string provided for fbool!"):
        meta_parse.fbool("")


def test_meta_parse_fint():
    assert meta_parse.fint(True) == 1
    assert meta_parse.fint("true") == 1
    assert meta_parse.fint("True") == 1
    assert meta_parse.fint(1) == 1
    assert meta_parse.fint(1.0) == 1
    assert meta_parse.fint("1") == 1

    assert meta_parse.fint(False) == 0
    assert meta_parse.fint("False") == 0
    assert meta_parse.fint("false") == 0
    assert meta_parse.fint("0") == 0
    assert meta_parse.fint("0.0") == 0
    assert meta_parse.fint(0) == 0
    assert meta_parse.fint(0.0) == 0

    assert meta_parse.fint("20") == 20
    assert meta_parse.fint(20.) == 20

    with pytest.raises(ValueError, match="Empty string provided for fint!"):
        meta_parse.fint("")


def test_meta_parse_fintlist():
    assert meta_parse.fintlist([True, 1, "1", 20.]) == [1, 1, 1, 20]
    assert meta_parse.fintlist("[True, 1, 1, 20.]") == [1, 1, 1, 20]


def test_meta_parse_lcstr():
    assert meta_parse.lcstr("PETER") == "peter"
    assert meta_parse.lcstr("Hans") == "hans"
