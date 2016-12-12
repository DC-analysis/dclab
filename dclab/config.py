#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
dclab configuration files and dicts
"""
from __future__ import division, print_function, unicode_literals

import codecs
import copy
from pkg_resources import resource_filename  # @UnresolvedImport
import sys

from . import definitions as dfn

if sys.version_info[0] == 2:
    string_classes = (str, unicode)
else:
    string_classes = str


def get_config_entry_choices(key, subkey, ignore_axes=[]):
    """ Returns the choices for a parameter, if any
    """
    ## Manually defined types:
    choices = []
    
    if key == "Plotting":
        if subkey == "KDE":
            choices = ["None", "Gauss", "Multivariate"]

        elif subkey in ["Axis X", "Axis Y"]:
            choices = copy.copy(dfn.uid)
            # remove unwanted axes
            for choice in ignore_axes:
                if choice in choices:
                    choices.remove(choice)
   
        elif subkey in ["Rows", "Columns"]:
            choices = [ str(i) for i in range(1,6) ]
        elif subkey in ["Scatter Marker Size"]:
            choices = [ str(i) for i in range(1,5) ]
        elif subkey.count("Scale "):
            choices = ["Linear", "Log"]
    return choices


def get_config_entry_dtype(key, subkey, cfg=None):
    """ Returns dtype of the parameter as defined in dclab.cfg
    """
    #default
    dtype = str

    ## Define dtypes and choices of cfg content
    # Iterate through cfg to determine standard dtypes
    # (also use cfg_init).    
    if cfg is None:
        cfg = cfg_init
    
    if key in cfg_init and subkey in cfg_init[key]:
        dtype = cfg_init[key][subkey].__class__
    else:
        try:
            dtype = cfg[key][subkey].__class__
        except KeyError:
            dtype = float

    return dtype


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
        cfg = {}
    
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
                    cfg[section] = dict()
                continue
            var, val = line.split("=", 1)
            var,val = map_config_value_str2type(var, val, capitalize=capitalize)
            if len(var) != 0 and len(str(val)) != 0:
                cfg[section][var] = val
    
    # 30µm channel?
    if ( "General" in cfg and
         not "Channel Width" in cfg["General"] and
         "Flow Rate [ul/s]" in cfg["General"] and
         cfg["General"]["Flow Rate [ul/s]"] >= 0.16     ):
        cfg["General"]["Channel Width"] = 30
    
    return cfg


def load_default_config():
    return load_config_file(cfgfile)


def map_config_value_str2type(var, val, capitalize=True):
    if not ( isinstance(val, string_classes) ):
        # already a type:
        return var.strip(), val
    var = var.strip()
    val = val.strip()
    if capitalize:
        # capitalize var
        if len(var) != 0:
            varsubs = var.split()
            newvar = u""
            for vs in varsubs:
                newvar += vs[0].capitalize()+vs[1:]+" "
            var = newvar.strip()
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


def map_config_value_type2str(var,val):
    var = var.strip()
    if len(var) != 0:
        if isinstance(val, list):
            out = "["
            for item in val:
                out += "{}, ".format(item)
            out = out.strip(" ,") + "]"
            return var, out
        elif isinstance(val, bool):
            return var, str(val)
        elif isinstance(val, str):
            return var, "'{}'".format(val.strip())
        elif isinstance(val, int):
            return var, "{:d}".format(val)
        elif isinstance(val, float):
            return var, "{:.12f}".format(val)
        else:
            return var, str(val)



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
            var, val = map_config_value_type2str(ikey, section[ikey])
            out.append("{} = {}".format(var,val))
        out.append("")
    
    with codecs.open(cfgfilename, "wb", "utf-8") as f:
        for i in range(len(out)):
            out[i] = out[i]+"\r\n"
        f.writelines(out)
    

def update_config_dict(oldcfg, newcfg):
    """ Update a configuration in librtdc format.
    
        
    Returns
    -------
    The new configuration, but it is also updated in-place.
    
    
    Notes
    -----
    Also converts from circularity to deformation in `newcfg`.
    """
    ## Defo to Circ conversion
    # new
    cmin = None
    cmax = None
    dmin = None
    dmax = None
    if "Filtering" in newcfg:
        if "Defo Max" in newcfg["Filtering"]:
            dmax = newcfg["Filtering"]["Defo Max"]
        if "Defo Min" in newcfg["Filtering"]:
            dmin = newcfg["Filtering"]["Defo Min"]
        if "Circ Max" in newcfg["Filtering"]:
            cmax = newcfg["Filtering"]["Circ Max"]
        if "Circ Min" in newcfg["Filtering"]:
            cmin = newcfg["Filtering"]["Circ Min"]
    # old
    cmino = None
    cmaxo = None
    dmino = None
    dmaxo = None
    if "Filtering" in oldcfg:
        if "Defo Max" in oldcfg["Filtering"]:
            dmaxo = oldcfg["Filtering"]["Defo Max"]
        if "Defo Min" in oldcfg["Filtering"]:
            dmino = oldcfg["Filtering"]["Defo Min"]
        if "Circ Max" in oldcfg["Filtering"]:
            cmaxo = oldcfg["Filtering"]["Circ Max"]
        if "Circ Min" in oldcfg["Filtering"]:
            cmino = oldcfg["Filtering"]["Circ Min"]
    # translation to new
    if cmin != cmino and cmin is not None:
        newcfg["Filtering"]["Defo Max"] = 1 - cmin
    if cmax != cmaxo and cmax is not None:
        newcfg["Filtering"]["Defo Min"] = 1 - cmax
    if dmin != dmino and dmin is not None:
        newcfg["Filtering"]["Circ Max"] = 1 - dmin
    if dmax != dmaxo and dmax is not None:
        newcfg["Filtering"]["Circ Min"] = 1 - dmax

    ## Contour
    if ("Plotting" in newcfg and
        "Contour Accuracy Circ" in newcfg["Plotting"] and
        not "Contour Accuracy Defo" in newcfg["Plotting"]):
        # If not contour accuracy for Defo is given, use that from Circ.
        newcfg["Plotting"]["Contour Accuracy Defo"] = newcfg["Plotting"]["Contour Accuracy Circ"]

    for key in list(newcfg.keys()):
        if not key in oldcfg:
            oldcfg[key] = dict()
        for skey in list(newcfg[key].keys()):
            oldcfg[key][skey] = newcfg[key][skey]

    ## Check missing values and set them to zero
    for item in dfn.uid:
        appends = [" Min", " Max"]
        for a in appends:
            if not item+a in oldcfg["Plotting"]:
                oldcfg["Plotting"][item+a] = 0
            if not item+a in oldcfg["Filtering"]:
                    oldcfg["Filtering"][item+a] = 0

    return oldcfg


### Load default configuration
cfgfile = resource_filename(__name__, 'dclab.cfg')
cfg = load_default_config()
cfg_init = copy.deepcopy(cfg)

