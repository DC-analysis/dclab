#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

import pathlib
import warnings

import numpy as np

from .compat import is_file_obj


class FilterIdExistsWarning(UserWarning):
    pass


class PolygonFilter(object):
    # Stuff that is done upon creation (not instantiation) of this class
    instances = []
    _instance_counter = 0

    def __init__(self, axes=None, points=None, inverted=False,
                 name=None, filename=None, fileid=0,
                 unique_id=None):
        """An object for filtering RTDC data based on a polygonial area

        Parameters
        ----------
        axes: tuple of str
            The axes/features on which the polygon is defined. The
            first axis is the x-axis. Example: ("area_um", "deform").
        points: array-like object of shape (N,2)
            The N coordinates (x,y) of the polygon. The exact order is
            important.
        inverted: bool
            Invert the polygon filter. This parameter is overridden
            if `filename` is given.
        name: str
            A name for the polygon (optional).
        filename : str
            A path to a .poly file as create by this classes' `save`
            method. If `filename` is given, all other parameters are
            ignored.
        fileid: int
            Which filter to import from the file (starting at 0).
        unique_id: int
            An integer defining the unique id of the new instance.

        Notes
        -----
        The minimal arguments to this class are either `filename` OR
        (`axes`, `points`). If `filename` is set, all parameters are
        taken from the given .poly file.
        """
        self.inverted = inverted
        # check if a filename was given
        if filename is not None:
            filename = pathlib.Path(filename)
            if not isinstance(fileid, int):
                raise ValueError("`fileid` must be an integer!")
            if not filename.exists():
                raise ValueError("Error, no such file: {}".format(filename))
            self.fileid = fileid
            # This also sets a unique id
            self._load(filename)
        else:
            self.axes = axes
            self.points = np.array(points, dtype=float)
            self.name = name
            if unique_id is None:
                # Force giving away a unique id
                unique_id = self._instance_counter

        # Set unique id
        if unique_id is not None:
            self._set_unique_id(unique_id)

        self._check_data()
        # if everything worked out, add to instances
        PolygonFilter.instances.append(self)

    def __eq__(self, pf):
        if (isinstance(pf, PolygonFilter) and
            self.inverted == pf.inverted and
            np.allclose(self.points, pf.points) and
                list(self.axes) == list(pf.axes)):
            eq = True
        else:
            eq = False
        return eq

    def _check_data(self):
        """Check if the data given is valid"""
        if self.axes is None:
            raise PolygonFilterError("`axes` parm not set.")
        if self.points is None:
            raise PolygonFilterError("`points` parm not set.")
        self.points = np.array(self.points)
        if self.points.shape[1] != 2:
            raise PolygonFilterError("data points' shape[1] must be 2.")
        if self.name is None:
            self.name = "polygon filter {}".format(self.unique_id)
        if not isinstance(self.inverted, bool):
            raise PolygonFilterError("`inverted` must be boolean.")

    def _load(self, filename):
        """Import all filters from a text file"""
        filename = pathlib.Path(filename)
        with filename.open() as fd:
            data = fd.readlines()

        # Get the strings that correspond to self.fileid
        bool_head = [l.strip().startswith("[") for l in data]

        int_head = np.squeeze(np.where(bool_head))
        int_head = np.atleast_1d(int_head)

        start = int_head[self.fileid]+1

        if len(int_head) > self.fileid+1:
            end = int_head[self.fileid+1]
        else:
            end = len(data)

        subdata = data[start:end]

        # separate all elements and strip them
        subdata = [[it.strip() for it in l.split("=")] for l in subdata]

        points = []

        for var, val in subdata:
            if var.lower() == "x axis":
                xaxis = val.lower()
            elif var.lower() == "y axis":
                yaxis = val.lower()
            elif var.lower() == "name":
                self.name = val
            elif var.lower() == "inverted":
                if val == "True":
                    self.inverted = True
            elif var.lower().startswith("point"):
                val = np.array(val.strip("[]").split(), dtype=float)
                points.append([int(var[5:]), val])
            else:
                raise KeyError("Unknown variable: {} = {}".
                               format(var, val))
        self.axes = (xaxis, yaxis)
        # sort points
        points.sort()
        # get only coordinates from points
        self.points = np.array([p[1] for p in points])

        # overwrite unique id
        unique_id = int(data[start-1].strip().strip("Polygon []"))
        self._set_unique_id(unique_id)

    def _set_unique_id(self, unique_id):
        """Define a unique id"""
        assert isinstance(unique_id, int), "unique_id must be an integer"

        if PolygonFilter.instace_exists(unique_id):
            newid = max(PolygonFilter._instance_counter, unique_id+1)
            msg = "PolygonFilter with unique_id '{}' exists.".format(unique_id)
            msg += " Using new unique id '{}'.".format(newid)
            warnings.warn(msg, FilterIdExistsWarning)
            unique_id = newid

        ic = max(PolygonFilter._instance_counter, unique_id+1)
        PolygonFilter._instance_counter = ic
        self.unique_id = unique_id

    @staticmethod
    def clear_all_filters():
        """Remove all filters and reset instance counter"""
        PolygonFilter.instances = []
        PolygonFilter._instance_counter = 0

    def copy(self, invert=False):
        """Return a copy of the current instance

        Parameters
        ----------
        invert: bool
            The copy will be inverted w.r.t. the original
        """
        if invert:
            inverted = not self.inverted
        else:
            inverted = self.inverted

        return PolygonFilter(axes=self.axes,
                             points=self.points,
                             name=self.name,
                             inverted=inverted)

    def filter(self, datax, datay):
        """Filter a set of datax and datay according to `self.points`"""
        f = np.ones(datax.shape, dtype=bool)
        for i, p in enumerate(zip(datax, datay)):
            f[i] = PolygonFilter.point_in_poly(p, self.points)

        if self.inverted:
            np.invert(f, f)

        return f

    @staticmethod
    def get_instance_from_id(unique_id):
        """Get an instance of the `PolygonFilter` using a unique id"""
        for instance in PolygonFilter.instances:
            if instance.unique_id == unique_id:
                return instance
        # if this does not work:
        raise KeyError("PolygonFilter with unique_id {} not found.".
                       format(unique_id))

    @staticmethod
    def import_all(path):
        """Import all polygons from a .poly file.

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
        """Determine whether an instance with this unique id exists"""
        try:
            PolygonFilter.get_instance_from_id(unique_id)
        except KeyError:
            return False
        else:
            return True

    @staticmethod
    def point_in_poly(p, poly):
        """Determine whether a point is within a polygon area

        Uses the ray casting algorithm.

        Parameters
        ----------
        p: float
            Coordinates of the point
        poly: array_like of shape (N, 2)
            Polygon (`PolygonFilter.points`)

        Returns
        -------
        inside: bool
            `True`, if point is inside.

        Notes
        -----
        If `p` lies on a side of the polygon, it is defined as

        - "inside" if it is on the top or right
        - "outside" if it is on the lower or left
        """
        poly = np.array(poly)
        n = poly.shape[0]
        inside = False
        x, y = p

        # Coarse bounding box exclusion:
        if (x <= poly[:, 0].max() and x > poly[:, 0].min()
                and y <= poly[:, 1].max() and y > poly[:, 1].min()):
            # The point is within the coarse bounding box.
            p1x, p1y = poly[0]  # point i in contour
            for ii in range(n):  # also covers (n-1, 0) (circular)
                p2x, p2y = poly[(ii+1) % n]  # point ii+1 in contour (circular)
                # Edge-wise fine bounding-ray exclusion.
                # Determine whether point is in the current ray,
                # defined by the y-range of p1 and p2 and whether
                # it is left of p1 and p2.
                if (y > min(p1y, p2y) and y <= max(p1y, p2y)  # in y-range
                        and x <= max(p1x, p2x)):  # left of p1 and p2
                    # Note that always p1y!=p2y due to the above test.
                    # Only Compute the x-coordinate of the intersection
                    # between line p1-p2 and the horizontal ray,
                    # ((y-p1y)*(p2x-p1x)/(p2y-p1y) + p1x),
                    # if x is not already known to be left of it
                    # (p1x==p2x in combination with x<=max(p1x, p2x) above).
                    if p1x == p2x or x <= (y-p1y)*(p2x-p1x)/(p2y-p1y) + p1x:
                        # Toggle `inside` if the ray intersects
                        # with the current edge.
                        inside = not inside
                # Move on to the next edge of the polygon.
                p1x, p1y = p2x, p2y

        return inside

    @staticmethod
    def remove(unique_id):
        """Remove a polygon filter from `PolygonFilter.instances`"""
        for p in PolygonFilter.instances:
            if p.unique_id == unique_id:
                PolygonFilter.instances.remove(p)

    def save(self, polyfile, ret_fobj=False):
        """Save all data to a text file (appends data if file exists).

        Polyfile can be either a path to a file or a file object that
        was opened with the write "w" parameter. By using the file
        object, multiple instances of this class can write their data.

        If `ret_fobj` is `True`, then the file object will not be
        closed and returned.
        """
        if is_file_obj(polyfile):
            fobj = polyfile
        else:
            fobj = pathlib.Path(polyfile).open("a")

        # Who the hell would use more then 10 million polygons or
        # polygon points? -> 08d (easier if other people want to import)
        data2write = []
        data2write.append("[Polygon {:08d}]".format(self.unique_id))
        data2write.append("X Axis = {}".format(self.axes[0]))
        data2write.append("Y Axis = {}".format(self.axes[1]))
        data2write.append("Name = {}".format(self.name))
        data2write.append("Inverted = {}".format(self.inverted))
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
        """Save all polygon filters"""
        nump = len(PolygonFilter.instances)
        if nump == 0:
            raise PolygonFilterError("There are not polygon filters to save.")
        for p in PolygonFilter.instances:
            # we return the ret_obj, so we don't need to open and
            # close the file multiple times.
            polyobj = p.save(polyfile, ret_fobj=True)
        polyobj.close()


class PolygonFilterError(BaseException):
    pass


def get_polygon_filter_names():
    """Get the names of all polygon filters in the order of creation"""
    names = []
    for p in PolygonFilter.instances:
        names.append(p.name)
    return names
