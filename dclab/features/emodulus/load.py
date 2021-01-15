import copy
import json
import pathlib
from pkg_resources import resource_filename

import numpy as np

from ... import definitions as dfn


#: Dictionary of look-up tables shipped with dclab.
INTERNAL_LUTS = {
    "FEM-2Daxis": "emodulus_lut.txt",
}


def load_lut(lut_data="FEM-2Daxis"):
    """Load LUT data from disk

    Parameters
    ----------
    lut_data: path, str, or tuple of (np.ndarray of shape (N, 3), dict)
        The LUT data to use. If it is a key in :const:`INTERNAL_LUTS`,
        then the respective LUT will be used. Otherwise, a path to a
        file on disk or a tuple (LUT array, meta data) is possible.

    Returns
    -------
    lut: np.ndarray of shape (N, 3)
        The LUT data for interpolation
    meta: dict
        The LUT metadata

    Notes
    -----
    If lut_data is a tuple of (lut, meta), then nothing is actually
    done (this is implemented for user convenience).
    """
    if isinstance(lut_data, tuple):
        lut, meta = lut_data
        lut = np.array(lut, copy=True)  # copy, because of normalization
        meta = copy.deepcopy(meta)  # copy, for the sake of consistency
    elif isinstance(lut_data, str) and lut_data in INTERNAL_LUTS:
        lut_path = resource_filename("dclab.features.emodulus",
                                     INTERNAL_LUTS[lut_data])
        lut, meta = load_mtext(lut_path)
    elif (isinstance(lut_data, (str, pathlib.Path))
          and pathlib.Path(lut_data).exists()):
        lut, meta = load_mtext(lut_data)
    else:
        raise ValueError("`name_path_arr` must be path, key, or array, "
                         "got '{}'!".format(lut_data))
    return lut, meta


def load_mtext(path):
    """Load colunm-based data from text files with metadata

    This file format is used for isoelasticity lines and look-up
    table data in dclab.

    The text file is loaded with `numpy.loadtxt`. The metadata
    are stored as a json string between the "BEGIN METADATA" and
    the "END METADATA" tags. The last comment (#) line before the
    actual data defines the features with units in square
    brackets and tab-separated. For instance:

        # [...]
        #
        # BEGIN METADATA
        # {
        #   "authors": "A. Mietke, C. Herold, J. Guck",
        #   "channel_width": 20.0,
        #   "channel_width_unit": "um",
        #   "date": "2018-01-30",
        #   "dimensionality": "2Daxis",
        #   "flow_rate": 0.04,
        #   "flow_rate_unit": "uL/s",
        #   "fluid_viscosity": 15.0,
        #   "fluid_viscosity_unit": "mPa s",
        #   "method": "analytical",
        #   "model": "linear elastic",
        #   "publication": "https://doi.org/10.1016/j.bpj.2015.09.006",
        #   "software": "custom Matlab code",
        #   "summary": "2D-axis-symmetric analytical solution"
        # }
        # END METADATA
        #
        # [...]
        #
        # area_um [um^2]    deform    emodulus [kPa]
        3.75331e+00    5.14496e-03    9.30000e-01
        4.90368e+00    6.72683e-03    9.30000e-01
        6.05279e+00    8.30946e-03    9.30000e-01
        7.20064e+00    9.89298e-03    9.30000e-01
        [...]
    """
    path = pathlib.Path(path).resolve()

    # Parse metadata
    size = path.stat().st_size
    dump = []
    injson = False
    prev_line = ""
    with path.open("r", errors='replace') as fd:
        while True:
            line = fd.readline()
            if fd.tell() == size:
                # something went wrong
                raise ValueError("EOF: Could not parse '{}'!".format(path))
            elif len(line.strip()) == 0:
                # ignore empty lines
                continue
            elif not line.strip().startswith("#"):
                # we are done here
                if prev_line == "":
                    raise ValueError("No column header in '{}'!".format(
                        path))
                break
            elif line.startswith("# BEGIN METADATA"):
                injson = True
                continue
            elif line.startswith("# END METADATA"):
                injson = False
            if injson:
                dump.append(line.strip("#").strip())
            else:
                # remember last line for header
                prev_line = line
    # metadata
    if dump:
        meta = json.loads("\n".join(dump))
    else:
        raise ValueError("No metadata json dump in '{}'!".format(path))
    # header
    feats = []
    units = []
    for hh in prev_line.strip("# ").split("\t"):
        if hh.count(" "):
            ft, un = hh.strip().split(" ")
            un = un.strip("[]")
        else:
            ft = hh
            un = ""
        if not dfn.scalar_feature_exists(ft):
            raise ValueError("Scalar feature not known: '{}'".format(ft))
        feats.append(ft)
        units.append(un)
    # data
    data = np.loadtxt(path)

    meta["column features"] = feats
    meta["column units"] = units

    # sanity checks
    assert meta["channel_width_unit"] == "um"
    assert meta["flow_rate_unit"] == "uL/s"
    assert meta["fluid_viscosity_unit"] == "mPa s"
    for ft, un in zip(feats, units):
        if ft == "deform":
            assert un == ""
        elif ft == "area_um":
            assert un == "um^2"
        elif ft == "emodulus":
            assert un == "kPa"
        elif ft == "volume":
            assert un == "um^3"
        else:
            assert False, "Please add sanity check for {}!".format(ft)

    # TODO:
    # - if everything works as expected, add "FEM" to valid methods
    #   and implement in Shape-Out 2
    if meta["method"] == "FEM":
        meta["method"] = "numerical"

    return data, meta
