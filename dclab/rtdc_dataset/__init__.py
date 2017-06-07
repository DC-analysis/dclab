#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals 

from .core import hashfile
from .config import Configuration
from . import fmt_dict, fmt_tdms, fmt_hierarchy


def RTDC_DataSet(tdms_path=None, ddict=None, hparent=None):
    kwinput = [tdms_path, ddict, hparent].count(None)
    assert kwinput==2, "Specify tdms_path OR ddict OR hparent"
    
    if tdms_path is not None:
        return fmt_tdms.RTDC_TDMS(tdms_path)
    elif ddict is not None:
        return fmt_dict.RTDC_Dict(ddict)
    elif hparent is not None:
        return fmt_hierarchy.RTDC_Hierarchy(hparent)
