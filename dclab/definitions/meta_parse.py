from __future__ import annotations

import numbers

import numpy as np
import numpy.typing as npt


def f1dfloatduple(value: list | tuple | npt.NDArray) -> tuple[float, float]:
    """Tuple of two floats (duple)"""
    if np.array(value).ndim != 1:
        raise ValueError(f"Value is not 1 dimensional, got {value}!")
    fvalue = tuple(float(i) for i in value)
    if len(fvalue) != 2:
        raise ValueError(f"Value must be of length two, "
                         f"got length {len(fvalue)}!")
    return fvalue


def f2dfloatarray(value: npt.ArrayLike) -> npt.NDArray:
    """numpy floating point array"""
    return np.array(value, dtype=np.float64)


def fbool(value: str | bool | float) -> bool:
    """boolean"""
    if isinstance(value, str):
        value = value.lower()
        if value == "false":
            return False
        elif value == "true":
            return True
        elif value:
            return bool(float(value))
        else:
            raise ValueError("Empty string provided for fbool!")
    else:
        return bool(float(value))


def fboolorfloat(value: str | bool | float) -> bool | float:
    """Bool or float"""
    if isinstance(value, (str, bool)) or value == 0:
        return fbool(value)
    elif isinstance(value, (int, float)):
        return float(value)
    else:
        raise ValueError(f"Value could not be converted to bool "
                         f"or float, got {value}!")


def fint(value: str | float) -> int:
    """integer"""
    if isinstance(value, str):
        # strings might have been saved wrongly as booleans
        value = value.lower()
        if value == "false":
            return 0
        elif value == "true":
            return 1
        elif value:
            return int(float(value))
        else:
            raise ValueError("Empty string provided for fint!")
    else:
        return int(float(value))


def fintlist(alist: str | list | tuple) -> list[int]:
    """A list of integers"""
    outlist = []
    if not isinstance(alist, (list, tuple)):
        # we have a string (comma-separated integers)
        alist = alist.strip().strip("[] ").split(",")
    for it in alist:
        if it:
            outlist.append(fint(it))
    return outlist


def lcstr(astr: str) -> str:
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
