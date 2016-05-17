#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
PolygonFilter classes and methods
"""
from __future__ import division, print_function, unicode_literals

import codecs
import numpy as np
import os
import sys


if sys.version_info[0] == 2:
    string_classes = (str, unicode)
else:
    string_classes = str


class PolygonFilter(object):
    """ An object for filtering RTDC data based on a polygonial area
    """
    # Stuff that is done upon creation (not instantiation) of this class
    instances = []
    _instance_counter = 0
        
    def __init__(self, axes=None, points=None, name=None,
                 filename=None, fileid=0, unique_id=None):
        """ Instantiates PolygonFilter
        
        Parameters
        ----------
        axes : tuple of str
            The axes on which the polygon is defined. The first axis is
            the x-axis. Example: ("Area", "Defo").
        points : array-like object of shape (N,2)
            The N coordinates (x,y) of the polygon. The exact order is
            important.
        name : str
            A name for the polygon (optional).
        filename : str (path)
            A path to a .poly file as create by this classes' `save`
            method. If filename is given, all other parameters are
            ignored.
        fileid : int
            Which filter to import from the file (starting at 0).
        unique_id : int
            An integer defining the unique id of the new instance.
        
        
        Notes
        -----
        The minimal arguments to this class are either `filename` OR
        (`axes`, `points`). If `filename` is set, all parameters are
        taken from the given .poly file.
        """
        # set our id and increment the _instance_counter
        if (unique_id is not None):
            if PolygonFilter.instace_exists(unique_id):
                raise ValueError("Instance with id {} already exists".
                                 format(unique_id))
            else:
                self.unique_id = unique_id
                PolygonFilter._instance_counter = max(
                           PolygonFilter._instance_counter, unique_id+1)
        else:
            self.unique_id = PolygonFilter._instance_counter
            PolygonFilter._instance_counter += 1

        # check if a filename was given
        if filename is not None:
            assert isinstance(fileid, int)
            assert os.path.exists(filename),\
                   "Error, no such file: {}".format(filename)
            self.fileid = fileid
            self._load(filename)

            # we are done here
        else:
            self.axes = axes
            self.points = np.array(points, dtype=float)
            self.name = name
        
        self._check_data()
        # if everything worked out, add to instances
        PolygonFilter.instances.append(self)
    
    
    def _check_data(self):
        """ Checks if the given data is valid.
        """
        assert self.axes is not None, "Error, `axes` parm not set."
        assert self.points is not None, "Error, `points` parm not set."
        self.points = np.array(self.points)
        assert self.points.shape[1] == 2, \
               "Error, data points must be have two coordinates."
        if self.name is None:
            self.name = "polygon filter {}".format(self.unique_id)
    
    @staticmethod
    def clear_all_filters():
        """ Removes all filters and resets instance counter.
        """
        PolygonFilter.instances = []
        PolygonFilter._instance_counter = 0
        
    
    def copy(self):
        """ Returns a copy of the current instance.
        """
        return PolygonFilter(axes=self.axes,
                             points=self.points,
                             name=self.name)
    
    
    def filter(self, datax, datay):
        """ Filters a set of datax and datay according to self.points.
        """
        f = np.ones(datax.shape, dtype=bool)
        for i, (x,y) in enumerate(zip(datax, datay)):
            f[i] = PolygonFilter.point_in_poly(x, y, self.points)
        return f
    
    
    @staticmethod
    def get_instance_from_id(unique_id):
        """ Returns the instance of the PolygonFilter with this
        unique_id.
        """
        for instance in PolygonFilter.instances:
            if instance.unique_id == unique_id:
                return instance
        # if this does not work:
        raise ValueError("PolygonFilter with unique_id {} not found.".
                         format(unique_id))
    
    @staticmethod
    def import_all(path):
        """ Imports all polygons from a .poly file.
        
        Returns a list of the imported polygon filters
        """
        plist = []
        fid = 0
        while True:
            try:
                p = PolygonFilter(filename=path, fileid=fid)
                plist.append(p)
                fid += 1
            except IndexError:
                break
        return plist
    
    @staticmethod
    def instace_exists(unique_id):
        """ Returns True if an instance with this id exists
        """
        try:
            PolygonFilter.get_instance_from_id(unique_id)
        except ValueError:
            return True
        else:
            return False
    
    def _load(self, filename):
        """ Imports all data from a text file.
        
        """
        fobj = codecs.open(filename, "r", "utf-8")
        data = fobj.readlines()
        fobj.close()

        # Get the strings that correspond to self.fileid
        bool_head = [ l.strip().startswith("[") for l in data ]
        
        int_head = np.squeeze(np.where(bool_head))
        int_head = np.atleast_1d(int_head)
        
        start = int_head[self.fileid]+1
        
        if len(int_head) > self.fileid+1:
            end = int_head[self.fileid+1]
        else:
            end = len(data)
        
        subdata = data[start:end]
        
        # separate all elements and strip them
        subdata = [ [ it.strip() for it in l.split("=") ] for l in subdata ]

        points = []
        
        for var, val in subdata:
            if var.lower() == "x axis":
                xaxis = val
            elif var.lower() == "y axis":
                yaxis = val
            elif var.lower() == "name":
                self.name = val
            elif len(var) == 0:
                pass
            elif var.lower().startswith("point"):
                val = np.array(val.strip("[]").split(), dtype=float)
                points.append([int(var[5:]), val])
            else:
                import IPython
                IPython.embed()
                raise NotImplementedError("Unknown variable: {} = {}".
                                          format(var, val))
        self.axes = (xaxis, yaxis)
        # sort points
        points.sort()
        # get only coordinates from points
        self.points = np.array([ p[1] for p in points ])
        
        # overwrite unique id
        self.unique_id = int(data[start-1].strip().strip("Polygon []"))

    @staticmethod
    def point_in_poly(x, y, poly):
        n = len(poly)
        inside = False

        p1x,p1y = poly[0]
        for i in range(n+1):
            p2x,p2y = poly[i % n]
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        if p1y != p2y:
                            xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x,p1y = p2x,p2y

        return inside

    @staticmethod
    def remove(unique_id):
        """ Removes a polygon filter with the unique_id from
            PolygonFilter.instances
        """
        for p in PolygonFilter.instances:
            if p.unique_id == unique_id:
                PolygonFilter.instances.remove(p)
        
    
    def save(self, polyfile, ret_fobj=False):
        """ Saves all data to a text file (appends data if file exists).
        
        Polyfile can be either a path to a file or a file object that
        was opened with the write "w" parameter. By using the file
        object, multiple instances of this class can write their data.
        
        If `ret_fobj` is `True`, then the file object will not be
        closed and returned. 
        """
        if isinstance(polyfile, string_classes):
            fobj = codecs.open(polyfile, "a", "utf-8")
        else:
            # file or tempfile._TemporaryFileWrapper
            fobj = polyfile
        # Who the hell would use more then 10 million polygons or
        # polygon points? -> 08d (easier if other people want to import)
        data2write = []
        data2write.append("[Polygon {:08d}]".format(self.unique_id))
        data2write.append("X Axis = {}".format(self.axes[0]))
        data2write.append("Y Axis = {}".format(self.axes[1]))
        data2write.append("Name = {}".format(self.name))
        for i, point in enumerate(self.points):
            data2write.append("point{:08d} = {:.15e} {:.15e}".format(i,
                                                            point[0],
                                                            point[1]))
        # Add new lines
        for i in range(len(data2write)):
            data2write[i] += "\n"

        # begin writing to fobj        
        fobj.writelines(data2write)
        
        if ret_fobj:
            return fobj
        else:
            fobj.close()
    
    @staticmethod
    def save_all(polyfile):
        """ Save all polygon filters
        """
        if len(PolygonFilter.instances) == 0:
            raise IndexError("There are not polygon filters to save.")
        for p in PolygonFilter.instances:
            # we return the ret_obj, so we don't need to open and
            # close the file multiple times.
            polyfile = p.save(polyfile, ret_fobj=True)
        polyfile.close()


def GetPolygonFilterNames():
    """ Returns the names of all polygon filters in the order of
        creation.
    """
    names = []
    for p in PolygonFilter.instances:
        names.append(p.name)
    return names

