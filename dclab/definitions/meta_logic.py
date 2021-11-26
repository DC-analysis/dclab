from . import feat_logic, meta_const, meta_parse


def config_key_exists(section, key):
    """Return `True` if the configuration key exists"""
    valid = False
    if section == "user":
        if isinstance(key, str) and key.strip():  # sanity check
            valid = True
    elif meta_const.config_funcs.get(section, {}).get(key, False):
        valid = True
    elif section == "online_filter":
        if key.endswith("soft limit") or key.endswith("polygon points"):
            # "online_filter:area_um,deform soft limit"
            # "online_filter:area_um,deform polygon points"
            f1, f2 = key.split(" ", 1)[0].split(",")
            valid = (feat_logic.feature_exists(f1)
                     and feat_logic.feature_exists(f2))
    return valid


def get_config_value_descr(section, key):
    """Return the description of a config value

    Returns `key` if not defined anywhere
    """
    descr = key
    if section == "user":
        pass
    elif meta_const.config_descr.get(section, {}).get(key, False):
        descr = meta_const.config_descr[section][key]
    elif section == "online_filter":
        if key.endswith("soft limit") or key.endswith("polygon points"):
            # "online_filter:area_um,deform soft limit"
            # "online_filter:area_um,deform polygon points"
            f1, f2 = key.split(" ", 1)[0].split(",")
            # remove the units with rsplit
            l1 = feat_logic.get_feature_label(f1).rsplit(" [", 1)[0]
            l2 = feat_logic.get_feature_label(f2).rsplit(" [", 1)[0]
            if key.endswith("soft limit"):
                descr = f"Soft limit, polygon ({l1}, {l2})"
            elif key.endswith("polygon points"):
                descr = f"Polygon ({l1}, {l2})"
    return descr


def get_config_value_func(section, key):
    """Return configuration type converter function"""
    func = None
    if section == "user":
        pass
    elif meta_const.config_funcs.get(section, {}).get(key, False):
        func = meta_const.config_funcs[section][key]
    elif section == "online_filter":
        if key.endswith("soft limit"):
            # "online_filter:area_um,deform soft limit"
            func = meta_parse.fbool
        elif key.endswith("polygon points"):
            # "online_filter:area_um,deform polygon points"
            func = meta_parse.f2dfloatarray

    if func is None:
        return lambda x: x
    else:
        return func


def get_config_value_type(section, key):
    """Return the expected type of a config value

    Returns `None` if no type is defined
    """
    typ = None
    if section == "user":
        pass
    elif meta_const.config_types.get(section, {}).get(key, False):
        typ = meta_const.config_types[section][key]
    elif section == "online_filter":
        if key.endswith("soft limit"):
            # "online_filter:area_um,deform soft limit"
            typ = meta_parse.func_types[meta_parse.fbool]
        elif key.endswith("polygon points"):
            typ = meta_parse.func_types[meta_parse.f2dfloatarray]
    return typ
