import numpy as np

from dclab.definitions import meta_logic, meta_parse


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
