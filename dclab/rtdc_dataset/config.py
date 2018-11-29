#!/usr/bin/python
# -*- coding: utf-8 -*-
"""RT-DC dataset configuration"""
from __future__ import division, print_function, unicode_literals

import copy
import pathlib

from ..compat import str_types
from .. import definitions as dfn


class CaseInsensitiveDict(dict):
    @classmethod
    def _k(cls, key):
        return key.lower() if isinstance(key, str_types) else key

    def __init__(self, *args, **kwargs):
        super(CaseInsensitiveDict, self).__init__(*args, **kwargs)
        self._convert_keys()

    def __getitem__(self, key):
        return super(CaseInsensitiveDict,
                     self).__getitem__(self.__class__._k(key))

    def __setitem__(self, key, value):
        super(CaseInsensitiveDict, self).__setitem__(
            self.__class__._k(key), value)

    def __delitem__(self, key):
        return super(CaseInsensitiveDict,
                     self).__delitem__(self.__class__._k(key))

    def __contains__(self, key):
        return super(CaseInsensitiveDict,
                     self).__contains__(self.__class__._k(key))

    def items(self):
        keys = list(self.keys())
        keys.sort()
        out = [(k, self[k]) for k in keys]
        return out

    def pop(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict,
                     self).pop(self.__class__._k(key), *args, **kwargs)

    def get(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict,
                     self).get(self.__class__._k(key), *args, **kwargs)

    def setdefault(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict,
                     self).setdefault(self.__class__._k(key), *args, **kwargs)

    def update(self, E={}, **F):
        super(CaseInsensitiveDict, self).update(self.__class__(E))
        super(CaseInsensitiveDict, self).update(self.__class__(**F))

    def _convert_keys(self):
        for k in list(self.keys()):
            v = super(CaseInsensitiveDict, self).pop(k)
            self.__setitem__(k, v)


class Configuration(object):
    def __init__(self, files=[], cfg={}):
        """Configuration class for RT-DC datasets

        This class has a dictionary-like interface to access
        and set configuration values, e.g.

        .. code::

            cfg = load_from_file("/path/to/config.txt")
            # access the channel width
            cfg["setup"]["channel width"]
            # modify the channel width
            cfg["setup"]["channel width"] = 30

        Parameters
        ----------
        files: list of files
            The config files with which to initialize the configuration
        cfg: dict-like
            The dictionary with which to initialize the configuration
        """
        self._cfg = CaseInsensitiveDict()

        # set initial default values
        self._init_default_values()

        # Update with additional dictionary
        self.update(cfg)

        # Load configuration files
        for f in files:
            self.update(load_from_file(f))

    def __contains__(self, key):
        return self._cfg.__contains__(key)

    def __getitem__(self, idx):
        if idx not in self and idx in dfn.config_keys:
            self._cfg[idx] = CaseInsensitiveDict()
        item = self._cfg.__getitem__(idx)
        if isinstance(item, str_types):
            item = item.lower()
        return item

    def __iter__(self):
        return self._cfg.__iter__()

    def __len__(self):
        return len(self._cfg)

    def __repr__(self):
        rep = ""
        keys = sorted(list(self.keys()))
        for key in keys:
            rep += "- {}\n".format(key)
            subkeys = sorted(list(self[key].keys()))
            for subkey in subkeys:
                rep += "   {}: {}\n".format(subkey, self[key][subkey])
        return rep

    def __setitem__(self, *args):
        self._cfg.__setitem__(*args)

    def _init_default_values(self):
        """Set default initial values

        The default values are hard-coded for backwards compatibility
        and for several functionalities in dclab.
        """
        # Do not filter out invalid event values
        self["filtering"]["remove invalid events"] = False
        # Enable filters switch is mandatory
        self["filtering"]["enable filters"] = True
        # Limit events integer to downsample output data
        self["filtering"]["limit events"] = 0
        # Polygon filter list
        self["filtering"]["polygon filters"] = []
        # Defaults to no hierarchy parent
        self["filtering"]["hierarchy parent"] = "none"
        # Check for missing min/max values and set them to zero
        for item in dfn.scalar_feature_names:
            appends = [" min", " max"]
            for a in appends:
                self["filtering"][item + a] = 0

    def copy(self):
        """Return copy of current configuration"""
        return Configuration(cfg=copy.deepcopy(self._cfg))

    def keys(self):
        """Return the configuration keys (sections)"""
        return self._cfg.keys()

    def save(self, filename):
        """Save the configuration to a file"""
        filename = pathlib.Path(filename)
        out = []
        keys = sorted(list(self.keys()))
        for key in keys:
            out.append("[{}]".format(key))
            section = self[key]
            ikeys = list(section.keys())
            ikeys.sort()
            for ikey in ikeys:
                var, val = keyval_typ2str(ikey, section[ikey])
                out.append("{} = {}".format(var, val))
            out.append("")

        with filename.open("w") as f:
            for i in range(len(out)):
                # win-like line endings
                out[i] = out[i]+"\n"
            f.writelines(out)

    def update(self, newcfg):
        """Update current config with a dictionary"""
        for key in newcfg.keys():
            if key not in self._cfg:
                self._cfg[key] = CaseInsensitiveDict()
            for skey in newcfg[key]:
                self._cfg[key][skey] = newcfg[key][skey]


def load_from_file(cfg_file):
    """Load the configuration from a file


    Parameters
    ----------
    cfg_file: str
        Path to configuration file


    Returns
    -------
    cfg : CaseInsensitiveDict
        Dictionary with configuration parameters
    """
    path = pathlib.Path(cfg_file).resolve()
    with path.open('r') as f:
        code = f.readlines()

    cfg = CaseInsensitiveDict()
    for line in code:
        # We deal with comments and empty lines
        # We need to check line length first and then we look for
        # a hash.
        line = line.split("#")[0].strip()
        if len(line) != 0:
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].lower()
                if section not in cfg:
                    cfg[section] = CaseInsensitiveDict()
                continue
            var, val = line.split("=", 1)
            var = var.strip().lower()
            val = val.strip("' ").strip('" ').strip()
            # convert parameter value to correct type
            if (section in dfn.config_funcs and
                    var in dfn.config_funcs[section]):
                # standard parameter with known type
                val = dfn.config_funcs[section][var](val)
            else:
                # unknown parameter (e.g. plotting in Shape-Out), guess type
                var, val = keyval_str2typ(var, val)
            if len(var) != 0 and len(str(val)) != 0:
                cfg[section][var] = val
    return cfg


def keyval_str2typ(var, val):
    """Convert a variable from a string to its correct type

    Parameters
    ----------
    var: str
        The variable name
    val: str
        The value of the variable represented as a string

    Returns
    -------
    varout: str
        Stripped lowercase `var`
    valout: any type
        The value converted from string to its presumed type

    Notes
    -----
    This method is heuristic and is only intended for usage in
    dclab.

    See Also
    --------
    keyval_typ2str: the opposite
    """
    if not (isinstance(val, str_types)):
        # already a type:
        return var.strip(), val
    var = var.strip().lower()
    val = val.strip()
    # Find values
    if len(var) != 0 and len(val) != 0:
        # check for float
        if val.startswith("[") and val.endswith("]"):
            if len(val.strip("[],")) == 0:
                # empty list
                values = []
            else:
                values = val.strip("[],").split(",")
                values = [float(v) for v in values]
            return var, values
        elif val.lower() in ["true", "y"]:
            return var, True
        elif val.lower() in ["false", "n"]:
            return var, False
        elif val[0] in ["'", '"'] and val[-1] in ["'", '"']:
            return var, val.strip("'").strip('"').strip()
        elif val in dfn.scalar_feature_names:
            return var, val
        else:
            try:
                return var, float(val.replace(",", "."))
            except ValueError:
                return var, val


def keyval_typ2str(var, val):
    """Convert a variable to a string

    Parameters
    ----------
    var: str
        The variable name
    val: any type
        The value of the variable

    Returns
    -------
    varout: str
        Stripped lowercase `var`
    valout: any type
        The value converted to a useful string representation

    See Also
    --------
    keyval_str2typ: the opposite
    """
    varout = var.strip()
    if isinstance(val, list):
        data = ", ".join([keyval_typ2str(var, it)[1] for it in val])
        valout = "["+data+"]"
    elif isinstance(val, float):
        valout = "{:.12f}".format(val)
    else:
        valout = "{}".format(val)
    return varout, valout
