"""RT-DC dataset core classes and methods"""
import abc
import random
import warnings

import numpy as np

from .. import definitions as dfn
from .. import downsampling
from ..polygon_filter import PolygonFilter
from .. import kde_methods

from .ancillaries import AncillaryFeature, FEATURES_RAPID
from .export import Export
from .filter import Filter


class LogTransformWarning(UserWarning):
    pass


class RTDCBase(abc.ABC):
    def __init__(self, identifier=None):
        """RT-DC measurement base class

        Notes
        -----
        Besides the filter arrays for each data feature, there is a manual
        boolean filter array ``RTDCBase.filter.manual`` that can be edited
        by the user - a boolean value of ``False`` means that the event is
        excluded from all computations.
        """
        #: Dataset format (derived from class name)
        self.format = self.__class__.__name__.split("_")[-1].lower()

        self._polygon_filter_ids = []
        # Events have the feature name as keys and contain nD ndarrays.
        self._events = {}
        # Ancillaries have the feature name as keys and a
        # tuple containing feature and hash as value.
        self._ancillaries = {}
        # Temporary features are defined by the user ad hoc at runtime.
        self._usertemp = {}
        #: Configuration of the measurement
        self.config = None
        #: Export functionalities; instance of
        #: :class:`dclab.rtdc_dataset.export.Export`.
        self.export = Export(self)
        # The filtering class is initialized with self._init_filters
        #: Filtering functionalities; instance of
        #: :class:`dclab.rtdc_dataset.filter.Filter`.
        self.filter = None
        #: Dictionary of log files. Each log file is a list of strings
        #: (one string per line).
        self.logs = {}
        #: Title of the measurement
        self.title = None
        #: Path or DCOR identifier of the dataset (set to "none"
        #: for :class:`RTDC_Dict`)
        self.path = None
        # Unique identifier
        if identifier is None:
            # Generate a unique identifier for this dataset
            rhex = [random.choice('0123456789abcdef') for _n in range(7)]
            self._identifier = "mm-{}_{}".format(self.format, "".join(rhex))
        else:
            self._identifier = identifier

    def __contains__(self, key):
        ct = False
        if key in self._events or key in self._usertemp:
            ct = True
        else:
            # Check ancillary features data
            if key in self._ancillaries:
                # already computed
                ct = True
            elif key in AncillaryFeature.feature_names:
                # get all instance of AncillaryFeature that
                # compute the feature `key`
                instlist = AncillaryFeature.get_instances(key)
                for inst in instlist:
                    if inst.is_available(self):
                        # to be computed
                        ct = True
                        break
        return ct

    def __getitem__(self, key):
        if key in self._events:
            return self._events[key]
        elif key in self._usertemp:
            return self._usertemp[key]
        # Try to find the feature in the ancillary features
        # (see ancillaries submodule for more information).
        # These features are cached in `self._ancillaries`.
        ancol = AncillaryFeature.available_features(self)
        if key in ancol:
            # The feature is available.
            anhash = ancol[key].hash(self)
            if (key in self._ancillaries and
                    self._ancillaries[key][0] == anhash):
                # Use cached value
                data = self._ancillaries[key][1]
            else:
                # Compute new value
                data_dict = ancol[key].compute(self)
                for okey in data_dict:
                    # Store computed value in `self._ancillaries`.
                    self._ancillaries[okey] = (anhash, data_dict[okey])
                data = data_dict[key]
            return data
        else:
            raise KeyError("Feature '{}' does not exist!".format(key))

    def __iter__(self):
        """An iterator over all valid scalar features"""
        mycols = []
        for col in self._feature_candidates:
            if col in self:
                mycols.append(col)
        mycols.sort()
        for col in mycols:
            yield col

    def __len__(self):
        keys = list(self._events.keys())
        keys.sort()
        for kk in keys:
            length = len(self._events[kk])
            if length:
                return length
        else:
            msg = "Could not determine size of dataset '{}'.".format(self)
            raise ValueError(msg)

    def __repr__(self):
        repre = "<{} '{}' at {}".format(self.__class__.__name__,
                                        self.identifier,
                                        hex(id(self)))
        if self.path != "none":
            repre += " ({})>".format(self.path)
        return repre

    @staticmethod
    def _apply_scale(a, scale, feat):
        """Helper function for transforming an aray to log-scale

        Parameters
        ----------
        a: np.ndarray
            Input array
        scale: str
            If set to "log", take the logarithm of `a`; if set to
            "linear" return `a` unchanged.
        feat: str
            Feature name (required for debugging)

        Returns
        -------
        b: np.ndarray
            The scaled array

        Notes
        -----
        If the scale is not "linear", then a new array is returned.
        All warnings are suppressed when computing `np.log(a)`, as
        `a` may have negative or nan values.
        """
        if scale == "linear":
            b = a
        elif scale == "log":
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                b = np.log(a)
                if len(w):
                    # Tell the user that the log-transformation issued
                    # a warning.
                    warnings.warn("Invalid values encounterd in np.log "
                                  "while scaling feature '{}'!".format(feat))
        else:
            raise ValueError("`scale` must be either 'linear' or 'log', "
                             + "got '{}'!".format(scale))
        return b

    @staticmethod
    def get_kde_spacing(a, scale="linear", method=kde_methods.bin_width_doane,
                        method_kw=None, feat="undefined", ret_scaled=False):
        """Convenience function for computing the contour spacing

        Parameters
        ----------
        a: ndarray
            feature data
        scale: str
            how the data should be scaled ("log" or "linear")
        method: callable
            KDE method to use (see `kde_methods` submodule)
        method_kw: dict
            keyword arguments to `method`
        feat: str
            feature name for debugging
        ret_scaled: bol
            whether or not to return the scaled array of `a`
        """
        if method_kw is None:
            method_kw = {}
        # Apply scale (no change for linear scale)
        asc = RTDCBase._apply_scale(a, scale, feat)
        # Apply multiplicator
        acc = method(asc, **method_kw)
        if ret_scaled:
            return acc, asc
        else:
            return acc

    @property
    def _feature_candidates(self):
        """List of feature candidates for this dataset

        Use with caution! Features in this list might not actually
        be available. Always check against `__contains__`.
        """
        feats = list(self._events.keys())
        feats += list(self._usertemp.keys())
        feats += list(AncillaryFeature.feature_names)
        feats = sorted(set(feats))
        # exclude non-standard features
        featsv = [ff for ff in feats if dfn.feature_exists(ff)]
        return featsv

    @property
    def _filter(self):
        """return the current filter boolean array"""
        warnings.warn("RTDCBase._filter is deprecated. Please use "
                      + "RTDCBase.filter.all instead.",
                      DeprecationWarning)
        return self.filter.all

    @property
    def _plot_filter(self):
        raise NotImplementedError(
            "RTDCBase._plot_filter has been removed in dclab 0.16.0. "
            + "Please use the output of RTDCBase.downsample_scatter "
            + "with the argument ret_mask instead.")

    def _init_filters(self):
        #: Filtering functionalities (this is an instance of
        #: :class:`dclab.rtdc_dataset.filter.Filter`.
        self.filter = Filter(self)
        self.reset_filter()

    @property
    def identifier(self):
        """Unique (unreproducible) identifier"""
        return self._identifier

    @property
    def features(self):
        """All available features"""
        mycols = []
        for col in self._feature_candidates:
            if col in self:
                mycols.append(col)
        mycols.sort()
        return mycols

    @property
    def features_innate(self):
        """All features excluding ancillary or temporary features"""
        innate = [ft for ft in self.features if ft in self._events]
        return innate

    @property
    def features_loaded(self):
        """All features that have been computed

        This includes ancillary features and temporary features.

        Notes
        -----
        Features that are computationally cheap to compute are
        always included. They are defined in
        :const:`dclab.rtdc_dataset.ancillaries.FEATURES_RAPID`.
        """
        features_innate = self.features_innate
        features_loaded = []
        for feat in self.features:
            if (feat in features_innate
                or feat in FEATURES_RAPID
                or feat in self._usertemp
                    or feat in self._ancillaries):
                # Note that there is no hash checking here for
                # ancillary features. This might be interesting
                # only in rare cases.
                features_loaded.append(feat)
        return features_loaded

    @property
    def features_scalar(self):
        """All scalar features available"""
        sclr = [ft for ft in self.features if dfn.scalar_feature_exists(ft)]
        return sclr

    @property
    @abc.abstractmethod
    def hash(self):
        """Reproducible dataset hash (defined by derived classes)"""

    def apply_filter(self, force=None):
        """Compute the filters for the dataset"""
        if force is None:
            force = []
        self.filter.update(rtdc_ds=self, force=force)

    def get_downsampled_scatter(self, xax="area_um", yax="deform",
                                downsample=0, xscale="linear",
                                yscale="linear", remove_invalid=False,
                                ret_mask=False):
        """Downsampling by removing points at dense locations

        Parameters
        ----------
        xax: str
            Identifier for x axis (e.g. "area_um", "aspect", "deform")
        yax: str
            Identifier for y axis
        downsample: int
            Number of points to draw in the down-sampled plot.
            This number is either

            - >=1: exactly downsample to this number by randomly adding
                   or removing points
            - 0  : do not perform downsampling
        xscale: str
            If set to "log", take the logarithm of the x-values before
            performing downsampling. This is useful when data are are
            displayed on a log-scale. Defaults to "linear".
        yscale: str
            See `xscale`.
        remove_invalid: bool
            Remove nan and inf values before downsampling; if set to
            `True`, the actual number of samples returned might be
            smaller than `downsample` due to infinite or nan values
            (e.g. due to logarithmic scaling).
        ret_mask: bool
            If set to `True`, returns a boolean array of length
            `len(self)` where `True` values identify the filtered
            data.

        Returns
        -------
        xnew, xnew: 1d ndarray of lenght `N`
            Filtered data; `N` is either identical to `downsample`
            or smaller (if `remove_invalid==True`)
        mask: 1d boolean array of length `len(RTDCBase)`
            Array for identifying the downsampled data points
        """
        if downsample < 0:
            raise ValueError("`downsample` must be zero or positive!")

        downsample = int(downsample)
        xax = xax.lower()
        yax = yax.lower()

        # Get data
        x = self[xax][self.filter.all]
        y = self[yax][self.filter.all]

        # Apply scale (no change for linear scale)
        xs = RTDCBase._apply_scale(x, xscale, xax)
        ys = RTDCBase._apply_scale(y, yscale, yax)

        _, _, idx = downsampling.downsample_grid(xs, ys,
                                                 samples=downsample,
                                                 remove_invalid=remove_invalid,
                                                 ret_idx=True)

        if ret_mask:
            # Mask is a boolean array of len(self)
            mask = np.zeros(len(self), dtype=bool)
            mids = np.where(self.filter.all)[0]
            mask[mids] = idx
            return x[idx], y[idx], mask
        else:
            return x[idx], y[idx]

    def get_kde_contour(self, xax="area_um", yax="deform", xacc=None,
                        yacc=None, kde_type="histogram", kde_kwargs=None,
                        xscale="linear", yscale="linear"):
        """Evaluate the kernel density estimate for contour plots

        Parameters
        ----------
        xax: str
            Identifier for X axis (e.g. "area_um", "aspect", "deform")
        yax: str
            Identifier for Y axis
        xacc: float
            Contour accuracy in x direction
        yacc: float
            Contour accuracy in y direction
        kde_type: str
            The KDE method to use
        kde_kwargs: dict
            Additional keyword arguments to the KDE method
        xscale: str
            If set to "log", take the logarithm of the x-values before
            computing the KDE. This is useful when data are are
            displayed on a log-scale. Defaults to "linear".
        yscale: str
            See `xscale`.

        Returns
        -------
        X, Y, Z : coordinates
            The kernel density Z evaluated on a rectangular grid (X,Y).
        """
        if kde_kwargs is None:
            kde_kwargs = {}
        xax = xax.lower()
        yax = yax.lower()
        kde_type = kde_type.lower()
        if kde_type not in kde_methods.methods:
            raise ValueError("Not a valid kde type: {}!".format(kde_type))

        # Get data
        x = self[xax][self.filter.all]
        y = self[yax][self.filter.all]

        xacc_sc, xs = RTDCBase.get_kde_spacing(
            a=x,
            feat=xax,
            scale=xscale,
            method=kde_methods.bin_width_doane,
            ret_scaled=True)

        yacc_sc, ys = RTDCBase.get_kde_spacing(
            a=y,
            feat=yax,
            scale=yscale,
            method=kde_methods.bin_width_doane,
            ret_scaled=True)

        if xacc is None:
            xacc = xacc_sc / 5

        if yacc is None:
            yacc = yacc_sc / 5

        # Ignore infs and nans
        bad = kde_methods.get_bad_vals(xs, ys)
        xc = xs[~bad]
        yc = ys[~bad]

        xnum = int(np.ceil((xc.max() - xc.min()) / xacc))
        ynum = int(np.ceil((yc.max() - yc.min()) / yacc))

        xlin = np.linspace(xc.min(), xc.max(), xnum, endpoint=True)
        ylin = np.linspace(yc.min(), yc.max(), ynum, endpoint=True)

        xmesh, ymesh = np.meshgrid(xlin, ylin, indexing="ij")

        kde_fct = kde_methods.methods[kde_type]
        if len(x):
            density = kde_fct(events_x=xs, events_y=ys,
                              xout=xmesh, yout=ymesh,
                              **kde_kwargs)
        else:
            density = np.array([])

        # Convert mesh back to linear scale if applicable
        if xscale == "log":
            xmesh = np.exp(xmesh)
        if yscale == "log":
            ymesh = np.exp(ymesh)

        return xmesh, ymesh, density

    def get_kde_scatter(self, xax="area_um", yax="deform", positions=None,
                        kde_type="histogram", kde_kwargs=None, xscale="linear",
                        yscale="linear"):
        """Evaluate the kernel density estimate for scatter plots

        Parameters
        ----------
        xax: str
            Identifier for X axis (e.g. "area_um", "aspect", "deform")
        yax: str
            Identifier for Y axis
        positions: list of two 1d ndarrays or ndarray of shape (2, N)
            The positions where the KDE will be computed. Note that
            the KDE estimate is computed from the points that
            are set in `self.filter.all`.
        kde_type: str
            The KDE method to use
        kde_kwargs: dict
            Additional keyword arguments to the KDE method
        xscale: str
            If set to "log", take the logarithm of the x-values before
            computing the KDE. This is useful when data are are
            displayed on a log-scale. Defaults to "linear".
        yscale: str
            See `xscale`.

        Returns
        -------
        density : 1d ndarray
            The kernel density evaluated for the filtered data points.
        """
        if kde_kwargs is None:
            kde_kwargs = {}
        xax = xax.lower()
        yax = yax.lower()
        kde_type = kde_type.lower()
        if kde_type not in kde_methods.methods:
            raise ValueError("Not a valid kde type: {}!".format(kde_type))

        # Get data
        x = self[xax][self.filter.all]
        y = self[yax][self.filter.all]

        # Apply scale (no change for linear scale)
        xs = RTDCBase._apply_scale(x, xscale, xax)
        ys = RTDCBase._apply_scale(y, yscale, yax)

        if positions is None:
            posx = None
            posy = None
        else:
            posx = RTDCBase._apply_scale(positions[0], xscale, xax)
            posy = RTDCBase._apply_scale(positions[1], yscale, yax)

        kde_fct = kde_methods.methods[kde_type]
        if len(x):
            density = kde_fct(events_x=xs, events_y=ys,
                              xout=posx, yout=posy,
                              **kde_kwargs)
        else:
            density = np.array([])

        return density

    def polygon_filter_add(self, filt):
        """Associate a Polygon Filter with this instance

        Parameters
        ----------
        filt: int or instance of `PolygonFilter`
            The polygon filter to add
        """
        if not isinstance(filt, (PolygonFilter, int, float)):
            msg = "`filt` must be a number or instance of PolygonFilter!"
            raise ValueError(msg)

        if isinstance(filt, PolygonFilter):
            uid = filt.unique_id
        else:
            uid = int(filt)
        # append item
        self.config["filtering"]["polygon filters"].append(uid)

    def polygon_filter_rm(self, filt):
        """Remove a polygon filter from this instance

        Parameters
        ----------
        filt: int or instance of `PolygonFilter`
            The polygon filter to remove
        """
        if not isinstance(filt, (PolygonFilter, int, float)):
            msg = "`filt` must be a number or instance of PolygonFilter!"
            raise ValueError(msg)

        if isinstance(filt, PolygonFilter):
            uid = filt.unique_id
        else:
            uid = int(filt)
        # remove item
        self.config["filtering"]["polygon filters"].remove(uid)

    def reset_filter(self):
        """Reset the current filter"""
        # reset filter instance
        self.filter.reset()
        # reset configuration
        # remember hierarchy parent
        hp = self.config["filtering"]["hierarchy parent"]
        self.config._init_default_filter_values()
        self.config["filtering"]["hierarchy parent"] = hp
