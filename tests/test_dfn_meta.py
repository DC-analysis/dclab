import numbers
import re
from inspect import getmembers, isfunction

import numpy as np

import pytest

from dclab.definitions import meta_logic, meta_parse


def assert_is_bool_false(value):
    assert isinstance(value, bool)
    assert not value


def assert_is_bool_true(value):
    assert isinstance(value, bool)
    assert value


@pytest.mark.parametrize("exists,sec,key", [
    [True, "user", "test"],
    [False, "user", 1],
    [True, "setup", "channel width"],
    [False, "setup", "funnel width"],
    [True, "online_filter", "area_um,deform soft limit"],
    [False, "online_filter", "peter,deform soft limit"],
    [True, "online_filter", "area_um,deform polygon points"],
    [False, "online_filter", "peter,deform polygon points"],
    [True, "online_filter", "pos_x max"],
    [True, "online_filter", "pos_x min"],
    [True, "online_filter", "pos_x soft limit"],
    [True, "online_filter", "fl1_max max"],
    [True, "online_filter", "fl2_max min"],
    [True, "online_filter", "fl3_max soft limit"],
    [True, "online_filter", "target duration"],
    [True, "online_filter", "area_ratio max"],
])
def test_meta_logic_config_key_exists(exists, sec, key):
    if exists:
        assert meta_logic.config_key_exists(sec, key)
    else:
        assert not meta_logic.config_key_exists(sec, key)


@pytest.mark.parametrize("sec,key,descr", [
    ["user", "key", "key"],
    ["experiment", "date", "Date of measurement ('YYYY-MM-DD')"],
    ["online_filter",
     "area_um,deform soft limit",
     "Soft limit, polygon (Area, Deformation)"],
    ["online_filter",
     "area_um,deform polygon points",
     "Polygon (Area, Deformation)"],
    ["online_filter", "fl1_max max", "Max. FL-1 maximum [a.u.]"],
    ["online_filter", "fl1_max min", "Min. FL-1 maximum [a.u.]"],
    ["online_filter",
     "area_ratio soft limit",
     "Soft limit, Porosity (convex to measured area ratio)"],
    ["online_filter", "fl3_max soft limit", "Soft limit, FL-3 maximum"],
    ["online_filter", "target duration", "Target measurement duration [min]"],
    ["online_filter", "pos_x max", "Max. Position along channel axis [µm]"],
])
def test_meta_logic_get_config_value_descr(sec, key, descr):
    assert meta_logic.get_config_value_descr(sec, key) == descr


@pytest.mark.parametrize("sec,key,func", [
    ["experiment", "date", str],
    ["online_filter", "area_um,deform soft limit", meta_parse.fbool],
    ["online_filter",
     "area_um,deform polygon points",
     meta_parse.f2dfloatarray],
])
def test_meta_logic_get_config_value_func(sec, key, func):
    assert meta_logic.get_config_value_func(sec, key) is func


def test_meta_logic_get_config_value_func_user():
    lamb = meta_logic.get_config_value_func("user", "key")
    assert lamb("peter") == "peter"


@pytest.mark.parametrize("sec,key,dtype", [
    ["user", "key", None],
    ["experiment", "date", str],
    ["online_filter", "area_um,deform soft limit", bool],
    ["online_filter",
     "area_um,deform polygon points",
     np.ndarray],
    ["online_filter", "fl1_max max", numbers.Number],
    ["online_filter", "pos_x min", numbers.Number],
])
def test_meta_logic_get_config_value_type(sec, key, dtype):
    this_type = meta_logic.get_config_value_type(sec, key)
    if isinstance(this_type, (tuple, list)):
        assert dtype in this_type
    else:
        assert dtype is this_type


def test_meta_parse_f2dfloatarray():
    assert meta_parse.f2dfloatarray([[1], [2]]).shape == (2, 1)


def test_meta_parse_f1dfloatduple():
    assert meta_parse.f1dfloatduple([1, 2]) == (1.0, 2.0)
    assert meta_parse.f1dfloatduple((8, 9)) == (8.0, 9.0)
    assert meta_parse.f1dfloatduple([-8.999, 0.332]) == (-8.999, 0.332)
    assert meta_parse.f1dfloatduple((-8.999, 0.332)) == (-8.999, 0.332)
    assert meta_parse.f1dfloatduple(np.array([3, -1])) == (3.0, -1.0)
    with pytest.raises(ValueError, match="Value must be of length two, "
                                         "got length 3!"):
        # the given value must be 1d
        assert meta_parse.f1dfloatduple((3, -1, 5.6))
    with pytest.raises(ValueError, match=re.escape(
            "Value is not 1 dimensional, "
            "got [[ 3.  -1. ]\n [ 4.   5.5]]!")):
        # the given value must be 1d
        assert meta_parse.f1dfloatduple(np.array([[3, -1], [4, 5.5]]))

    # check the func_types mapping
    typ = meta_parse.func_types[meta_parse.f1dfloatduple]
    assert isinstance((1.0, 2.0), typ)


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


def test_meta_parse_fboolorfloat():
    """Check fboolorfloat format"""
    assert meta_parse.fboolorfloat(True) is True
    assert meta_parse.fboolorfloat(False) is False
    assert meta_parse.fboolorfloat("true") is True
    assert meta_parse.fboolorfloat("False") is False

    assert meta_parse.fboolorfloat(2.0) == 2.0
    assert meta_parse.fboolorfloat(-0.44) == -0.44

    assert meta_parse.fboolorfloat(1) == 1.0
    assert meta_parse.fboolorfloat(-42) == -42.0

    with pytest.raises(ValueError, match=re.escape(
            "Value could not be converted to bool or float, got [1, 2, 3]!")):
        assert meta_parse.fboolorfloat([1, 2, 3])

    # check the func_types mapping
    typ = meta_parse.func_types[meta_parse.fboolorfloat]
    assert isinstance(True, typ)
    assert isinstance(2.0, typ)


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


def test_meta_parse_function_mapping():
    """Each function should have type(s) mapping in `meta_parse.func_types`"""

    meta_parse_funcs = getmembers(meta_parse, isfunction)
    type_mapping = list(meta_parse.func_types.keys())

    for meta_parse_func in meta_parse_funcs:
        assert meta_parse_func[1] in type_mapping, (
            f"The `meta_parse` module function `{meta_parse_func}` does not "
            "have a type mapping in meta_parse.func_types.")
