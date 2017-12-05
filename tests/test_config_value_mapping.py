#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from dclab.rtdc_dataset import config


def test_map_typ2str():
    assert config.keyval_typ2str("a", True)[1] == "True"
    assert config.keyval_typ2str("a", [1, 2, 3])[1] == "[1, 2, 3]"
    assert config.keyval_typ2str("a", "astring")[1] == "astring"
    assert config.keyval_typ2str("a", 120)[1] == "120"
    assert config.keyval_typ2str("a", 1.0)[1] == "1.000000000000"
    assert config.keyval_typ2str("a", 1j)[1] == "1j"


def test_map_str2typ():
    assert config.keyval_str2typ(" a ", 1) == ("a", 1)
    assert config.keyval_str2typ("a", "[1, 2, 3]")[1] == [1, 2, 3]
    assert not config.keyval_str2typ("a", "False")[1]
    assert not config.keyval_str2typ("a", "false")[1]
    assert config.keyval_str2typ("a", "true")[1]
    assert config.keyval_str2typ("area_um", "is this case even used?")[
        1] == "is this case even used?"
    assert config.keyval_str2typ("a", "1,true")[1] == "1,true"


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
