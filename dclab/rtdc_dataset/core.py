"""RT-DC dataset core classes and methods"""
import abc
import hashlib
import os.path
import pathlib
import traceback
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


class FeatureShouldExistButNotFoundWarning(UserWarning):
    pass


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
        #: Local basins are basins that are defined on the user's file system.
        #: For reasons of data security (leaking data from a server or from a
        #: user's file system), dclab only allows remote basins (see
        #: :func:`basins_retrieve`) by default. This variable is set to True
        #: for the RTDC_HDF5 file format, because it implies the data are
        #: located on the user's computer.
        self._local_basins_allowed = False

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
        # List of basin identifiers that should be ignored, used to
        # avoid circular basin dependencies
        self._basins_ignored = []
        # List of all features available via basins
        self._basins_features = None
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
        # 2. Check for h5dataset-based, file-based, or other basin data,
        #    in that order.
        for basin_type in ["internal", "file", None]:
            data = self._get_basin_feature_data(feat, basin_type=basin_type)
            if data is not None:
                return data
        # 3. Check for ancillary features that can be computed
        data = self._get_ancillary_feature_data(feat)
        if data is not None:
            return data
        if feat in self:
            warnings.warn(f"The feature {feat} is supposedly defined in "
                          f"{self}, but I cannot get its data. Please "
                          f"make sure you have not defined any unreachable "
                          f"remote basins.",
                          FeatureShouldExistButNotFoundWarning)
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
        anhash = None
        if feat in AncillaryFeature.feature_names:
            # Try to find the feature in the ancillary features
            # (see feat_anc_core submodule for more information).
            # These features are cached in `self._ancillaries`.
            ancol = AncillaryFeature.available_features(self)
            if feat in ancol:
                # The feature is generally available.
                if feat in self._ancillaries:
                    # We have already computed the feature. Make sure that we
                    # have the updated one by checking the hash.
                    anhash = ancol[feat].hash(self)
                    if self._ancillaries[feat][0] == anhash:
                        # Use cached value
                        data = self._ancillaries[feat][1]
                # We either already have the ancillary feature or have to
                # compute it. We only compute it if we are asked to.
                if data is None and not no_compute:
                    anhash = anhash or ancol[feat].hash(self)
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
            basin_type: Literal["file", "internal", "remote", None] = None):
        """Return feature data from basins

        Parameters
        ----------
        feat: str
            Name of the feature
        basin_type: str or bool
            The basin type to look at, which is either "file"-based
            (e.g. local on disk), "remote"-based (e.g. S3), or
            "internal"-type (e.g. h5py.Dataset inside the current HDF5 file).
            Defaults to `None` which means no preference.

        Returns
        -------
        data:
            The feature object (array-like) or None if it could not
            be found or was not computed.
        """
        data = None
        if self.basins:
            for bn in list(self.basins):
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
                except (KeyError, OSError, PermissionError):
                    # Basin data not available
                    pass
                except feat_basin.BasinNotAvailableError:
                    # remove the basin from the list
                    # TODO:
                    #  Check whether this has an actual effect. It could be
                    #  that due to some iterative process `self`
                    #  gets re-initialized and we have to go through this
                    #  again.
                    self._basins.remove(bn)
                    warnings.warn(
                        f"Removed unavailable basin {bn} from {self}")
                except BaseException:
                    warnings.warn(f"Could not access {feat} in {self}:\n"
                                  f"{traceback.format_exc()}")
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
        features = []
        for col in self._feature_candidates:
            if col in self:
                features.append(col)
        features.sort()
        return features

    @property
    def features_ancillary(self):
        """All available ancillary features

        This includes all ancillary features, excluding the features
        that are already in `self.features_innate`. This means that
        there may be overlap between `features_ancillary` and e.g.
        `self.features_basin`.

        .. versionadded:: 0.58.0

        """
        features_innate = self.features_innate
        features_ancillary = []
        for feat in AncillaryFeature.feature_names:
            if feat not in features_innate and feat in self:
                features_ancillary.append(feat)
        return sorted(features_ancillary)

    @property
    def features_basin(self):
        """All features accessed via upstream basins from other locations"""
        if self._basins_features is None:
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
                self._basins_features = sorted(set(features))
            else:
                self._basins_features = []
        return self._basins_features

    @property
    def features_innate(self):
        """All features excluding ancillary, basin, or temporary features"""
        innate = [ft for ft in self.features if ft in self._events]
        return innate

    @property
    def features_loaded(self):
        """All features that have been computed

        This includes ancillary features and temporary features.

        Notes
        -----
        Ancillary features that are computationally cheap to compute are
        always included. They are defined in
        :const:`dclab.rtdc_dataset.feat_anc_core.FEATURES_RAPID`.
        """
        features_loaded = self.features_local + self.features_innate
        features_loaded += [f for f in self.features if f in FEATURES_RAPID]
        return sorted(set(features_loaded))

    @property
    def features_local(self):
        """All features that are, with certainty, really fast to access

        Local features is a slimmed down version of `features_loaded`.
        Nothing needs to be computed, not even rapid features
        (:const:`dclab.rtdc_dataset.feat_anc_core.FEATURES_RAPID`).
        And features from remote sources that have not been downloaded
        already are excluded. Ancillary and temporary features that are
        available are included.
        """
        features_local = []
        # Note that the hierarchy format just calls its hparent's
        # `features_local`.
        if hasattr(self._events, "_cached_events"):
            features_local += list(self._events._cached_events.keys())

        if self.format == "hdf5":
            features_local += list(self._events.keys())

        # Get into the basins.
        for bn in self.basins:
            if (bn.basin_format == "hdf5"
                    and bn.basin_type == "file"
                    and bn.is_available()):
                features_local += bn.ds.features_local
            elif bn._ds is not None:
                features_local += bn.ds.features_local

        # If they are here, then we use them:
        features_local += list(self._ancillaries.keys())
        features_local += list(self._usertemp.keys())

        return sorted(set(features_local))

    @property
    def features_scalar(self):
        """All scalar features available"""
        sclr = [ft for ft in self.features if dfn.scalar_feature_exists(ft)]
        return sclr

    @property
    @abc.abstractmethod
    def hash(self):
        """Reproducible dataset hash (defined by derived classes)"""

    def ignore_basins(self, basin_identifiers):
        """Ignore these basin identifiers when looking for features

        This is used to avoid circular basin dependencies.
        """
        self._basins_ignored += basin_identifiers

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

        if xacc is None or xacc == 0:
            xacc = xacc_sc / 5

        if yacc is None or yacc == 0:
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

        .. versionchanged:: 0.57.5

            "file"-type basins are only available for subclasses that
            set the `_local_basins_allowed` attribute to True.
        """
        basins = []
        bc = feat_basin.get_basin_classes()
        # Sort basins according to priority
        bdicts_srt = sorted(self.basins_get_dicts(),
                            key=feat_basin.basin_priority_sorted_key)
        bd_keys = [bd["key"] for bd in bdicts_srt if "key" in bd]
        bd_keys += self._basins_ignored
        for bdict in bdicts_srt:
            if bdict["format"] not in bc:
                warnings.warn(f"Encountered unsupported basin "
                              f"format '{bdict['format']}'!")
                continue
            if "key" in bdict and bdict["key"] in self._basins_ignored:
                warnings.warn(
                    f"Encountered cyclic basin dependency '{bdict['key']}'",
                    feat_basin.CyclicBasinDependencyFoundWarning)
                continue

            # Basin initialization keyword arguments
            kwargs = {
                "name": bdict.get("name"),
                "description": bdict.get("description"),
                # Honor features intended by basin creator.
                "features": bdict.get("features"),
                # Which mapping we are using ("same", "basinmap1", ...)
                "mapping": bdict.get("mapping", "same"),
                # For non-identical mapping ("basinmap1", etc.), we
                # need the referring dataset.
                "mapping_referrer": self,
                # Make sure the measurement identifier is checked.
                "measurement_identifier": self.get_measurement_identifier(),
                # allow to ignore basins
                "ignored_basins": bd_keys,
            }

            # Check whether this basin is supported and exists
            if bdict["type"] == "internal":
                b_cls = bc[bdict["format"]]
                bna = b_cls(bdict["paths"][0], **kwargs)
                # In contrast to file-type basins, we just add all remote
                # basins without checking first. We do not check for
                # the availability of remote basins, because they could
                # be temporarily inaccessible (unstable network connection)
                # and because checking the availability of remote basins
                # normally takes a lot of time.
                basins.append(bna)
            elif bdict["type"] == "file":
                if not self._local_basins_allowed:
                    warnings.warn(f"Basin type 'file' not allowed for format "
                                  f"'{self.format}'")
                    # stop processing this basin
                    continue
                p_paths = list(bdict["paths"])
                # translate Windows and Unix relative paths
                for pi in list(p_paths):  # [sic] create a copy of the list
                    if pi.count(".."):
                        if pi[2:].count("/") and os.path.sep == r"\\":
                            # Windows
                            p_paths.append(pi.replace("/", r"\\"))
                        elif pi[2:].count(r"\\") and os.path.sep == "/":
                            # Unix
                            p_paths.append(pi.replace(r"\\", "/"))
                # perform the actual check
                for pp in p_paths:
                    pp = pathlib.Path(pp)
                    # Instantiate the proper basin class
                    b_cls = bc[bdict["format"]]
                    # Try absolute path
                    bna = b_cls(pp, **kwargs)
                    if bna.verify_basin():
                        basins.append(bna)
                        break
                    # Try relative path
                    this_path = pathlib.Path(self.path)
                    if this_path.exists():
                        # Insert relative path
                        bnr = b_cls(this_path.parent / pp, **kwargs)
                        if bnr.verify_basin():
                            basins.append(bnr)
                            break
            elif bdict["type"] == "remote":
                for url in bdict["urls"]:
                    # Instantiate the proper basin class
                    b_cls = bc[bdict["format"]]
                    bna = b_cls(url, **kwargs)
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
