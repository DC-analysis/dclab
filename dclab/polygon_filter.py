
import io
import pathlib
import warnings

import numpy as np

from .external.skimage.measure import points_in_poly
from .util import hashobj


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
        self._points = None
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
            if len(axes) != 2:
                raise ValueError("`axes` must have length 2, "
                                 + "got '{}'!".format(axes))
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

    def __getstate__(self):
        state = {
            "axis x": self.axes[0],
            "axis y": self.axes[1],
            "identifier": self.unique_id,
            "inverted": self.inverted,
            "name": self.name,
            "points": self.points.tolist()
        }
        return state

    def __setstate__(self, state):
        if state["identifier"] != self.unique_id:
            raise ValueError("Polygon filter identifier mismatch!")
        self.axes = [state["axis x"], state["axis y"]]
        self.inverted = state["inverted"]
        self.name = state["name"]
        self.points = state["points"]

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
        with filename.open("r", errors="replace") as fd:
            data = fd.readlines()

        # Get the strings that correspond to self.fileid
        bool_head = [li.strip().startswith("[") for li in data]

        int_head = np.squeeze(np.where(bool_head))
        int_head = np.atleast_1d(int_head)

        start = int_head[self.fileid]+1

        if len(int_head) > self.fileid+1:
            end = int_head[self.fileid+1]
        else:
            end = len(data)

        subdata = data[start:end]

        # separate all elements and strip them
        subdata = [[it.strip() for it in li.split("=")] for li in subdata]

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

    @property
    def hash(self):
        """Hash of `axes`, `points`, and `inverted`"""
        return hashobj([self.axes, self.points, self.inverted])

    @property
    def points(self):
        # make sure points always is an array (so we can use .tobytes())
        return np.array(self._points)

    @points.setter
    def points(self, points):
        self._points = points

    @staticmethod
    def clear_all_filters():
        """Remove all filters and reset instance counter"""
        PolygonFilter.instances = []
        PolygonFilter._instance_counter = 0

    @staticmethod
    def unique_id_exists(pid):
        """Whether or not a filter with this unique id exists"""
        for instance in PolygonFilter.instances:
            if instance.unique_id == pid:
                exists = True
                break
        else:
            exists = False
        return exists

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
        points = np.concatenate([datax.reshape(-1, 1),
                                 datay.reshape(-1, 1)],
                                axis=1)
        f = points_in_poly(points=points, verts=self.points)

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
        p: tuple of floats
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

        - "inside" if it is on the lower or left
        - "outside" if it is on the top or right

        .. versionchanged:: 0.24.1
            The new version uses the cython implementation from
            scikit-image. In the old version, the inside/outside
            definition was the other way around. In favor of not
            having to modify upstram code, the scikit-image
            version was adapted.
        """
        points = np.array(p).reshape(1, 2)
        f = points_in_poly(points=points, verts=np.array(poly))
        return f.item()

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
        if isinstance(polyfile, io.IOBase):
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
        if len(PolygonFilter.instances) == 0:
            raise PolygonFilterError("There are no polygon filters to save.")
        for p in PolygonFilter.instances:
            # we return the ret_obj, so we don't need to open and
            # close the file multiple times.
            polyobj = p.save(polyfile, ret_fobj=True)
        # close the object after we are done saving all filters
        polyobj.close()


class PolygonFilterError(BaseException):
    pass


def get_polygon_filter_names():
    """Get the names of all polygon filters in the order of creation"""
    names = []
    for p in PolygonFilter.instances:
        names.append(p.name)
    return names
