import numbers

import numpy as np


def f1dfloatduple(value):
    """Tuple of two floats (duple)"""
    if np.array(value).ndim != 1:
        raise ValueError(f"Value is not 1 dimensional, got {value}!")
    value = tuple(float(i) for i in value)
    if len(value) != 2:
        raise ValueError(f"Value must be of length two, "
                         f"got length {len(value)}!")
    return value


def f2dfloatarray(value):
    """numpy floating point array"""
    return np.array(value, dtype=np.float64)


def fbool(value):
    """boolean"""
    if isinstance(value, str):
        value = value.lower()
        if value == "false":
            value = False
        elif value == "true":
            value = True
        elif value:
            value = bool(float(value))
        else:
            raise ValueError("Empty string provided for fbool!")
    else:
        value = bool(float(value))
    return value


def fboolorfloat(value):
    """Bool or float"""
    if isinstance(value, (str, bool)) or value == 0:
        return fbool(value)
    elif isinstance(value, (int, float)):
        return float(value)
    else:
        raise ValueError(f"Value could not be converted to bool "
                         f"or float, got {value}!")


def fint(value):
    """integer"""
    if isinstance(value, str):
        # strings might have been saved wrongly as booleans
        value = value.lower()
        if value == "false":
            value = 0
        elif value == "true":
            value = 1
        elif value:
            value = int(float(value))
        else:
            raise ValueError("Empty string provided for fint!")
    else:
        value = int(float(value))
    return value


def fintlist(alist):
    """A list of integers"""
    outlist = []
    if not isinstance(alist, (list, tuple)):
        # we have a string (comma-separated integers)
        alist = alist.strip().strip("[] ").split(",")
    for it in alist:
        if it:
            outlist.append(fint(it))
    return outlist


def lcstr(astr):
    """lower-case string"""
    return astr.lower()


#: maps functions to their expected output types
func_types = {
    f1dfloatduple: (tuple, np.ndarray),
    f2dfloatarray: np.ndarray,
    fbool: (bool, np.bool_),
    fboolorfloat: (bool, np.bool_, float),
    fint: numbers.Integral,
    fintlist: list,
    float: numbers.Number,
    lcstr: str}
