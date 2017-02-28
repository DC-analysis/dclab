#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
dclab configuration files and dicts
"""
from __future__ import division, print_function, unicode_literals

import codecs
import sys

from .rtdc_dataset.config import CaseInsensitiveDict, keyval_str2typ

if sys.version_info[0] == 2:
    string_classes = (str, unicode)
else:
    string_classes = str



def load_config_file(cfgfilename, cfg=None, capitalize=True):
    """ Load a configuration file
    
    
    Parameters
    ----------
    cfgfilename : absolute path
        Filename of the configuration
    cfg : dict
        Dictionary to update/overwrite. If `cfg` is not set, a new
        dicitonary will be created.
    capitalize : bool
        Capitalize dictionary entries. This is useful to
        prevent duplicate entries in configurations dictionaries
        like "Flow Rate" and "flow rate". It is not useful when
        dictionary entries are not capitalized, e.g. for other
        configuration files like index files of ShapeOut sessions.
        
    
    Returns
    -------
    cfg : dict
        Dictionary with configuration in librtdc format.
        
    
    Notes
    -----
    If a [General] section is loaded and if the keyword "Channel Width"
    is not defined and if the flow rate is >= 0.16µl/s, then we
    set the Channel Width to 30 µm.
    """
    if cfg is None:
        cfg = CaseInsensitiveDict()

    with codecs.open(cfgfilename, 'r', 'utf-8') as f:
        code = f.readlines()
    
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
            var,val = keyval_str2typ(var, val)
            if len(var) != 0 and len(str(val)) != 0:
                cfg[section][var] = val
    
    # 30µm channel?
    if ( "General" in cfg and
         not "Channel Width" in cfg["General"] and
         "Flow Rate [ul/s]" in cfg["General"] and
         cfg["General"]["Flow Rate [ul/s]"] >= 0.16     ):
        cfg["General"]["Channel Width"] = 30

    return cfg


def save_config_file(cfgfilename, cfg):
    """ Save configuration to text file


    Parameters
    ----------
    cfgfilename : absolute path
        Filename of the configuration
    cfg : dict
        Dictionary containing configuration.

    """
    out = []
    keys = list(cfg.keys())
    keys.sort()
    for key in keys:
        out.append("[{}]".format(key))
        section = cfg[key]
        ikeys = list(section.keys())
        ikeys.sort()
        for ikey in ikeys:
            out.append("{} = {}".format(ikey,section[ikey]))
        out.append("")
    
    with codecs.open(cfgfilename, "wb", "utf-8") as f:
        for i in range(len(out)):
            out[i] = out[i]+"\r\n"
        f.writelines(out)
