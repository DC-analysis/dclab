#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

import sys

if sys.version_info[0] == 2:
    str_types = (str, unicode)
else:
    str_types = str


class Bool(object):
    """A boolean object"""
    def __init__(self, value):
        if isinstance(value, str_types):
            value = value.lower()
            if value == "false":
                value = False
            elif value == "true":
                value = True
            elif value:
                value = bool(float(value))
            else:
                raise ValueError("empty string")
        else:
            value = bool(float(value))
        self.value = value

    def __repr__(self):
        return "{}".format(self.value)

    def __bool__(self):
        return self.value
    
    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)
    
    def __len__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, Bool):
            return self.value == other.value
        else:
            return self.value == other


class IntList(list):
    """A list of integers"""
    def __init__(self, alist=[]):
        super(IntList, self).__init__()
        if not isinstance(alist, (list, tuple)):
            # we have a string (comma-separated integers)
            alist = alist.strip().strip("[] ").split(",")
        for it in alist:
            if it:
                self.append(it)

    def append(self, value):
        super(IntList, self).append(int(value))
