#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
RTDC_DataSet configuration
"""
from __future__ import division, print_function, unicode_literals

import codecs
import copy
import numpy as np
from pkg_resources import resource_filename
import sys


from .. import definitions as dfn


if sys.version_info[0] == 2:
    string_classes = (str, unicode)
else:
    string_classes = str



class CaseInsensitiveDict(dict):
    @classmethod
    def _k(cls, key):
        return key.lower() if isinstance(key, string_classes) else key
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
        """Configuration of an RTDC_DataSet
        
        Parameters
        ----------
        files: list of files
            The config files with which to initialize the configuration
        cfg: dict-like
            The dictionary with which to initialize the configuration
        """
        self._cfg = CaseInsensitiveDict()
        
        # Load default configuration
        self.update(default)
        # Load configuration files
        for f in files:
            self.update(load_from_file(f))
        # Update with additional dictionary
        self.update(cfg)

        self._fix_config()
        if rtdc_ds is not None:
            self._complete_config_from_rtdc_ds(rtdc_ds)


    def __contains__(self, key):
        return self._cfg.__contains__(key)


    def __getitem__(self, idx):
        return self._cfg.__getitem__(idx)


    def __setitem__(self, *args):
        self._cfg.__setitem__(*args)


    def _fix_config(self):
        """Fix missing config keys using default values
        
        These are conditional default values that complete the
        static values in `config_default.cfg`.
        """
        ## Old RTDC data files did not mention channel width for high flow rates
        assert "general" in self._cfg, "Configuration not properly initialized!"
        if not "flow Rate [ul/s]" in self._cfg["general"]:
            self._cfg["general"]["flow Rate [ul/s]"] = np.nan
        if not "channel width" in self._cfg["general"]:
            if self._cfg["general"]["flow Rate [ul/s]"] < 0.16:
                self._cfg["general"]["channel width"] = 20
            else:
                self._cfg["general"]["channel width"] = 30
        ## Check for missing min/max values and set them to zero
        for item in dfn.uid:
            appends = [" min", " max"]
            for a in appends:
                if not item+a in self._cfg["plotting"]:
                    self._cfg["plotting"][item+a] = 0
                if not item+a in self._cfg["filtering"]:
                    self._cfg["filtering"][item+a] = 0


    def _complete_config_from_rtdc_ds(self, rtdc_ds):
        """Complete configuration using data columns from RTDC_DataSet"""
        ## Sensible values for default contour accuracies
        keys = []
        for prop in dfn.rdv:
            if not np.allclose(getattr(rtdc_ds, prop), 0):
                # There are values for this uid
                keys.append(prop)
        # This lambda function seems to do a good job
        accl = lambda a: (np.nanmax(a)-np.nanmin(a))/10
        defs = [["contour accuracy {}", accl],
                ["kde multivariate {}", accl],
               ]
        pltng = self._cfg["plotting"]
        for k in keys:
            for d, l in defs:
                var = d.format(dfn.cfgmap[k])
                if not var in pltng:
                    pltng[var] = l(getattr(rtdc_ds, k))
        # Update data size
        self._cfg["general"]["cell number"] = rtdc_ds.time.shape[0]


    def copy(self):
        """Return copy of current configuration"""
        return Configuration(cfg=copy.deepcopy(self._cfg))


    def keys(self):
        return self._cfg.keys()


    def update(self, newcfg):
        """Update current config with new dictionary"""
        if isinstance(newcfg, Configuration):
            newcfg = newcfg._cfg
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
    with codecs.open(cfg_file, 'r', 'utf-8') as f:
        code = f.readlines()

    cfg = CaseInsensitiveDict()
    for line in code:
        # We deal with comments and empty lines
        # We need to check line length first and then we look for
        # a hash.
        line = line.split("#")[0].strip().lower()
        if len(line) != 0:
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                if not section in cfg:
                    cfg[section] = CaseInsensitiveDict()
                continue
            var, val = line.split("=", 1)
            var, val = map_config_value_str2type(var, val)
            if len(var) != 0 and len(str(val)) != 0:
                cfg[section][var] = val
    return cfg



def map_config_value_str2type(var, val):
    if not ( isinstance(val, string_classes) ):
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
        elif val in dfn.uid:
            return var, val
        else:
            try:
                return var, float(val.replace(",","."))
            except ValueError:
                return var, val


default_file = resource_filename(__name__, 'config_default.cfg')
default = load_from_file(default_file)