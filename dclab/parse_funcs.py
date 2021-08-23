import numbers

import numpy as np


def f2dfloatarray(value):
    return np.array(value, dtype=float)


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
            raise ValueError("empty string")
    else:
        value = bool(float(value))
    return value


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
            raise ValueError("empty string")
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
    f2dfloatarray: np.ndarray,
    fbool: (bool, np.bool_),
    fint: numbers.Integral,
    fintlist: list,
    float: numbers.Number,
    lcstr: str}
