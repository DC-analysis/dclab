#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import codecs
import numpy as np
import os
from os.path import abspath, dirname, join
import shutil
import sys
import warnings
import zipfile


# Add parent directory to beginning of path variable
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from dclab import config


def test_map_typ2str():
    assert config.map_config_value_type2str("a", True)[1] == "True"
    assert config.map_config_value_type2str("a", [1, 2, 3])[1] == "[1, 2, 3]"
    assert config.map_config_value_type2str("a", "astring")[1] == "'astring'"
    assert config.map_config_value_type2str("a", 120)[1] == "120"
    assert config.map_config_value_type2str("a", 1.0)[1] == "1.000000000000"
    assert config.map_config_value_type2str("a", 1j)[1] == "1j"


def test_map_str2typ():
    assert config.map_config_value_str2type(" a ", 1) == ("a", 1)
    assert config.map_config_value_str2type("a", "[1, 2, 3]")[1] == [1, 2, 3]
    assert config.map_config_value_str2type("a", "False")[1] == False
    assert config.map_config_value_str2type("a", "false")[1] == False
    assert config.map_config_value_str2type("a", "true")[1] == True
    assert config.map_config_value_str2type("Area", "is this case even used?")[1] == "is this case even used?"
    assert config.map_config_value_str2type("a", "1,true")[1] == "1,true"


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
