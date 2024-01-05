"""RT-DC dataset core classes and methods"""
import abc
import hashlib
import pathlib
from typing import Literal
import uuid
import random
import warnings

import numpy as np

from .. import definitions as dfn
from .. import downsampling
from ..polygon_filter import PolygonFilter
from .. import kde_methods

from .feat_anc_core import AncillaryFeature, FEATURES_RAPID
from . import feat_basin
from .export import Export
from .filter import Filter


class LogTransformWarning(UserWarning):
    pass


class RTDCBase(abc.ABC):
    def __init__(self, identifier=None, enable_basins=True):
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

        # Cache attribute used for __len__()-function
        self._length = None
        self._polygon_filter_ids = []
        # Events have the feature name as keys and contain nD ndarrays.
        self._events = {}
        # Ancillaries have the feature name as keys and a
        # tuple containing feature and hash as value.
        self._ancillaries = {}
        # Temporary features are defined by the user ad hoc at runtime.
        self._usertemp = {}
        # List of :class:`.Basin` for external features
        self._basins = None
        #: Configuration of the measurement
        self.config = None
        #: Export functionalities; instance of
        #: :class:`dclab.rtdc_dataset.export.Export`.
        self.export = Export(self)
        # Filtering functionalities; instance of
        # :class:`dclab.rtdc_dataset.filter.Filter`.
        self._ds_filter = None
        #: Dictionary of log files. Each log file is a list of strings
        #: (one string per line).
        self.logs = {}
        #: Dictionary of tables. Each table is an indexable compound numpy
        #: array.
        self.tables = {}
        #: Title of the measurement
        self.title = None
        #: Path or DCOR identifier of the dataset (set to "none"
        #: for :class:`RTDC_Dict`)
        self.path = None
        # Unique, random identifier
        if identifier is None:
            # Generate a unique, random identifier for this dataset
            rhex = [random.choice('0123456789abcdef') for _n in range(7)]
            self._identifier = "mm-{}_{}".format(self.format, "".join(rhex))
        else:
            self._identifier = identifier

        # Basins are initialized in the "basins" property function
        self._enable_basins = enable_basins

    def __contains__(self, feat):
        ct = False
        if (feat in self._events
                or feat in self._usertemp
                or feat in self.features_basin):
            ct = True
        else:
            # Check ancillary features data
            if feat in self._ancillaries:
                # already computed
                ct = True
            elif feat in AncillaryFeature.feature_names:
                # get all instance of AncillaryFeature that
                # check availability of the feature `feat`
                instlist = AncillaryFeature.get_instances(feat)
                for inst in instlist:
                    if inst.is_available(self):
                        # to be computed
                        ct = True
                        break
        return ct

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def __getitem__(self, feat):
        if feat in self._events:
            return self._events[feat]
        elif feat in self._usertemp:
            return self._usertemp[feat]
        # 1. Check for cached ancillary data
        data = self._get_ancillary_feature_data(feat, no_compute=True)
        if data is not None:
            return data
        # 2. Check for file-based basin data
        data = self._get_basin_feature_data(feat, basin_type="file")
        if data is not None:
            return data
        # 3. Check for other basin data
        data = self._get_basin_feature_data(feat)
        if data is not None:
            return data
        # 4. Check for ancillary features that can be computed
        data = self._get_ancillary_feature_data(feat)
        if data is not None:
            return data
        if feat in self:
            warnings.warn(f"The feature {feat} is supposedly defined in "
                          f"{self}, but I cannot get its data. Please "
                          f"make sure you have not defined any unreachable "
                          f"remote basins.",
                          UserWarning)
        # Not here ¯\_(ツ)_/¯
        raise KeyError(f"Feature '{feat}' does not exist in {self}!")

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
        if self._length is None:
            self._length = self._get_length()
        return self._length

    def _get_length(self):
        # Try to get length from metadata.
        length = self.config["experiment"].get("event count")
        if length is not None:
            return length
        # Try to get the length from the feature sizes
        keys = list(self._events.keys()) or self.features_basin
        keys.sort()
        for kk in keys:
            length = len(self[kk])
            if length:
                return length
        else:
            raise ValueError(f"Could not determine size of dataset '{self}'.")

    def __repr__(self):
        repre = "<{} '{}' at {}".format(self.__class__.__name__,
                                        self.identifier,
                                        hex(id(self)))
        if self.path != "none":
            repre += " ({})>".format(self.path)
        return repre

    @property
    def basins(self):
        """Basins containing upstream features from other datasets"""
        if self._basins is None:
            if self._enable_basins:
                self._basins = self.basins_retrieve()
            else:
                self._basins = []
        return self._basins

    @property
    def filter(self):
        """Filtering functionalities; instance of :class:`.Filter`"""
        self._assert_filter()
        return self._ds_filter

    def _assert_filter(self):
        if self._ds_filter is None:
            self._ds_filter = Filter(self)

    def _get_ancillary_feature_data(self,
                                    feat: str,
                                    no_compute: bool = False):
        """Return feature data of ancillary features

        Parameters
        ----------
        feat: str
            Name of the feature
        no_compute: bool
            Whether to bother computing the feature. If it is not
            already computed, return None instead

        Returns
        -------
        data:
            The feature object (array-like) or None if it could not
            be found or was not computed.
        """
        data = None
        # Try to find the feature in the ancillary features
        # (see feat_anc_core submodule for more information).
        # These features are cached in `self._ancillaries`.
        ancol = AncillaryFeature.available_features(self)
        if feat in ancol:
            # The feature is available.
            anhash = ancol[feat].hash(self)
            if (feat in self._ancillaries and
                    self._ancillaries[feat][0] == anhash):
                # Use cached value
                data = self._ancillaries[feat][1]
            elif not no_compute:
                # Compute new value
                data_dict = ancol[feat].compute(self)
                for okey in data_dict:
                    # Store computed value in `self._ancillaries`.
                    self._ancillaries[okey] = (anhash, data_dict[okey])
                data = data_dict[feat]
        return data

    def _get_basin_feature_data(
            self,
            feat: str,
            basin_type: Literal["file", "remote", None] = None):
        """Return feature data from basins

        Parameters
        ----------
        feat: str
            Name of the feature
        basin_type: str or bool
            The basin type to look at, which is either "file"-based
            (e.g. local on disk), "remote"-based (e.g. S3) all
            basins (None, default).

        Returns
        -------
        data:
            The feature object (array-like) or None if it could not
            be found or was not computed.
        """
        data = None
        if self.basins:
            for bn in self.basins:
                if basin_type is not None and basin_type != bn.basin_type:
                    # User asked for specific basin type
                    continue
                try:
                    # There are all kinds of errors that may happen here.
                    # Note that `bn.features` can already trigger an
                    # availability check that may raise a ValueError.
                    # TODO:
                    #  Introduce some kind of callback so the user knows
                    #  why the data are not available. The current solution
                    #  (fail silently) is not sufficiently transparent,
                    #  especially when considering networking issues.
                    if feat in bn.features:
                        data = bn.get_feature_data(feat)
                        # The data are available, we may abort the search.
                        break
                except BaseException:
                    # Basin data not available
                    pass
        return data

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
        ret_scaled: bool
            whether to return the scaled array of `a`
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
        feats += self.features_basin
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
    def features_basin(self):
        """All features accessed via upstream basins from other locations"""
        if self.basins:
            features = []
            for bn in self.basins:
                if bn.features and set(bn.features) <= set(features):
                    # We already have the features from a different basin.
                    # There might be a basin availability check going on
                    # somewhere, but we are not interested in it.
                    continue
                if bn.is_available():
                    features += bn.features
            return sorted(set(features))
        else:
            return []

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
        :const:`dclab.rtdc_dataset.feat_anc_core.FEATURES_RAPID`.
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

    def close(self):
        """Close any open files or connections, including basins

        If implemented in a subclass, the subclass must call this
        method via `super`, otherwise basins are not closed. The
        subclass is responsible for closing its specific file handles.
        """
        if self._basins:
            for bn in self._basins:
                bn.close()

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
            computing the KDE. This is useful when data are
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
            The KDE method to use, see :const:`.kde_methods.methods`
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

    def basins_get_dicts(self):
        """Return the list of dictionaries describing the dataset's basins"""
        # Only implement this for classes that support this
        return []

    def basins_retrieve(self):
        """Load all basins available

        .. versionadded:: 0.54.0

        In dclab 0.51.0, we introduced basins, a simple way of combining
        HDF5-based datasets (including the :class:`.HDF5_S3` format).
        The idea is to be able to store parts of the dataset
        (e.g. images) in a separate file that could then be located
        someplace else (e.g. an S3 object store).

        If an RT-DC file has "basins" defined, then these are sought out and
        made available via the `features_basin` property.
        """
        basins = []
        bc = feat_basin.get_basin_classes()
        muid = self.get_measurement_identifier()
        # Sort basins according to priority
        bdicts_srt = sorted(self.basins_get_dicts(),
                            key=feat_basin.basin_priority_sorted_key)
        for bdict in bdicts_srt:
            if bdict["format"] not in bc:
                warnings.warn(f"Encountered unsupported basin "
                              f"format '{bdict['format']}'!")
                continue
            # Check whether this basin is supported and exists
            kwargs = {
                "name": bdict.get("name"),
                "description": bdict.get("description"),
                # Honor features intended by basin creator.
                "features": bdict.get("features"),
                # Make sure the measurement identifier is checked.
                "measurement_identifier": self.get_measurement_identifier(),
            }

            if bdict["type"] == "file":
                for pp in bdict["paths"]:
                    pp = pathlib.Path(pp)
                    # Instantiate the proper basin class
                    bcls = bc[bdict["format"]]
                    # Try absolute path
                    bna = bcls(pp, **kwargs)
                    if (bna.is_available()
                            and bna.get_measurement_identifier() == muid):
                        basins.append(bna)
                        break
                    # Try relative path
                    thispath = pathlib.Path(self.path)
                    if thispath.exists():
                        # Insert relative path
                        bnr = bcls(thispath.parent / pp, **kwargs)
                        if (bnr.is_available()
                                and bnr.get_measurement_identifier() == muid):
                            basins.append(bnr)
                            break
            elif bdict["type"] == "remote":
                for url in bdict["urls"]:
                    # Instantiate the proper basin class
                    bcls = bc[bdict["format"]]
                    bna = bcls(url, **kwargs)
                    # In contrast to file-type basins, we just add all remote
                    # basins without checking first. We do not check for
                    # the availability of remote basins, because they could
                    # be temporarily inaccessible (unstable network connection)
                    # and because checking the availability of remote basins
                    # normally takes a lot of time.
                    basins.append(bna)
            else:
                warnings.warn(
                    f"Encountered unsupported basin type '{bdict['type']}'!")
        return basins

    def get_measurement_identifier(self):
        """Return a unique measurement identifier

        Return the [experiment]:"run identifier" configuration feat, if it
        exists. Otherwise, return the MD5 sum computed from the measurement
        time, date, and setup identifier.

        Returns `None` if no identifier could be found or computed.

        .. versionadded:: 0.51.0

        """
        identifier = self.config.get("experiment", {}).get("run identifier",
                                                           None)
        if identifier is None:
            time = self.config.get("experiment", {}).get("time", None)
            date = self.config.get("experiment", {}).get("date", None)
            sid = self.config.get("setup", {}).get("identifier", None)
            if None not in [time, date, sid]:
                # only compute an identifier if all of the above are defined.
                hasher = hashlib.md5(f"{time}_{date}_{sid}".encode("utf-8"))
                identifier = str(uuid.UUID(hex=hasher.hexdigest()))
        return identifier

    def polygon_filter_add(self, filt):
        """Associate a Polygon Filter with this instance

        Parameters
        ----------
        filt: int or instance of `PolygonFilter`
            The polygon filter to add
        """
        self._assert_filter()  # [sic] initialize the filter if not done yet
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
