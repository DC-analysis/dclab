#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
RT-DC dataset configuration
"""
from __future__ import division, print_function, unicode_literals

import copy
import io
import numpy as np
import sys

from .. import definitions as dfn


if sys.version_info[0] == 2:
    str_types = (str, unicode)
else:
    str_types = str



class CaseInsensitiveDict(dict):
    @classmethod
    def _k(cls, key):
        return key.lower() if isinstance(key, str_types) else key
    def __init__(self, *args, **kwargs):
        super(CaseInsensitiveDict, self).__init__(*args, **kwargs)
        self._convert_keys()
    def __getitem__(self, key):
        return super(CaseInsensitiveDict, self).__getitem__(self.__class__._k(key))
    def __setitem__(self, key, value):
        super(CaseInsensitiveDict, self).__setitem__(self.__class__._k(key), value)
    def __delitem__(self, key):
        return super(CaseInsensitiveDict, self).__delitem__(self.__class__._k(key))
    def __contains__(self, key):
        return super(CaseInsensitiveDict, self).__contains__(self.__class__._k(key))
    def has_key(self, key):
        return super(CaseInsensitiveDict, self).has_key(self.__class__._k(key))
    def items(self):
        keys = list(self.keys())
        keys.sort()
        out = [(k,self[k]) for k in keys]
        return out
    def pop(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).pop(self.__class__._k(key), *args, **kwargs)
    def get(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).get(self.__class__._k(key), *args, **kwargs)
    def setdefault(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).setdefault(self.__class__._k(key), *args, **kwargs)
    def update(self, E={}, **F):
        super(CaseInsensitiveDict, self).update(self.__class__(E))
        super(CaseInsensitiveDict, self).update(self.__class__(**F))
    def _convert_keys(self):
        for k in list(self.keys()):
            v = super(CaseInsensitiveDict, self).pop(k)
            self.__setitem__(k, v)



class Configuration(object):
    def __init__(self, files=[], cfg={}, rtdc_ds=None):
        """Configuration of an RT-DC dataset
        
        Parameters
        ----------
        files: list of files
            The config files with which to initialize the configuration
        cfg: dict-like
            The dictionary with which to initialize the configuration
        rtdc_ds: instance of RTDCBase
            An RT-DC data set.
        """
        self._cfg = CaseInsensitiveDict()

        # Update with additional dictionary
        self.update(cfg)

        # Load configuration files
        for f in files:
            self.update(load_from_file(f))

        self._fix_config()
        # Complete configuration settings
        if rtdc_ds is not None:
            self._complete_config_from_rtdc_ds(rtdc_ds)


    def __contains__(self, key):
        return self._cfg.__contains__(key)


    def __getitem__(self, idx):
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
        keys = self.keys()
        keys.sort()
        for key in keys:
            rep += "- {}\n".format(key)
            subkeys = self[key].keys()
            subkeys.sort()
            for subkey in subkeys:
                rep += "   {}: {}\n".format(subkey, self[key][subkey])
        return rep


    def __setitem__(self, *args):
        self._cfg.__setitem__(*args)


    def _fix_config(self):
        """Fix missing config keys using default values
        
        The default values are hard-coded for backwards compatibility
        and for several functionalities in dclab.
        """
        ## Filtering
        if not "filtering" in self:
            self["filtering"] = CaseInsensitiveDict()
        # Do not filter out invalid event values
        if not "remove invalid events" in self["filtering"]:
            self["filtering"]["remove invalid events"] = False
        # Enable filters switch is mandatory
        if not "enable filters" in self["filtering"]:
            self["filtering"]["enable filters"] = True
        # Limit events integer to downsample output data
        if not "limit events" in self["filtering"]:
            self["filtering"]["limit events"] = 0
        # Polygon filter list
        if not "polygon filters" in self["filtering"]:
            self["filtering"]["polygon filters"] = []
        # Defaults to no hierarchy parent
        if not "hierarchy parent" in self["filtering"]:
            self["filtering"]["hierarchy parent"] = "none"
        # Check for missing min/max values and set them to zero
        for item in dfn.column_names:
            appends = [" min", " max"]
            for a in appends:
                if not item+a in self["filtering"]:
                    self["filtering"][item+a] = 0
        ## General
        if not "general" in self:
            self["general"] = CaseInsensitiveDict()
        # Old RTDC data files have an offset in the recorded video file
        if not "video frame offset" in self["general"]:
            self["general"]["video frame offset"] = 1
        # Old RTDC data files did not mention channel width for high flow rates
        if not "flow Rate [ul/s]" in self["general"]:
            self["general"]["flow Rate [ul/s]"] = np.nan
        if not "channel width" in self["general"]:
            if self["general"]["flow Rate [ul/s]"] < 0.16:
                self["general"]["channel width"] = 20
            else:
                self["general"]["channel width"] = 30
        ## Image
        if not "image" in self:
            self["image"] = CaseInsensitiveDict()
        # Old RTDC data files do not have resolution
        if not "pix size" in self["image"]:
            self["image"]["pix size"] = 0.34


    def _complete_config_from_rtdc_ds(self, rtdc_ds):
        """Complete configuration using data columns from RT-DC dataset"""
        # Update data size
        self["general"]["cell number"] = len(rtdc_ds)


    def copy(self):
        """Return copy of current configuration"""
        return Configuration(cfg=copy.deepcopy(self._cfg))


    def keys(self):
        return self._cfg.keys()


    def save(self, filename):
        """Save the configuration to a file"""
        out = []
        keys = list(self.keys())
        keys.sort()
        for key in keys:
            out.append("[{}]".format(key))
            section = self[key]
            ikeys = list(section.keys())
            ikeys.sort()
            for ikey in ikeys:
                var, val = keyval_typ2str(ikey, section[ikey])
                out.append("{} = {}".format(var, val))
            out.append("")
        
        with io.open(filename, "w") as f:
            for i in range(len(out)):
                # win-like line endings
                out[i] = out[i]+"\r\n"
            f.writelines(out)


    def update(self, newcfg):
        """Update current config with new dictionary"""
        for key in newcfg.keys():
            if not key in self._cfg:
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
    with io.open(cfg_file, 'r') as f:
        code = f.readlines()

    cfg = CaseInsensitiveDict()
    for line in code:
        # We deal with comments and empty lines
        # We need to check line length first and then we look for
        # a hash.
        line = line.split("#")[0].strip()
        if len(line) != 0:
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                if not section in cfg:
                    cfg[section] = CaseInsensitiveDict()
                continue
            var, val = line.split("=", 1)
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
    if not ( isinstance(val, str_types) ):
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
        elif val in dfn.column_names:
            return var, val
        else:
            try:
                return var, float(val.replace(",","."))
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
