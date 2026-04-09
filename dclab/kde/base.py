import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI

from ..cached import umbrella_cache
from ..definitions import feat_const
from .binning import bin_width_percentile
from .contours import find_contours_level, get_quantile_levels
from .helpers import get_bad_vals
from .methods import methods


def kde_handler(kde):
    return [
        # uniquely identifies an RTDC file, hierarchy child, or online data
        kde.rtdc_ds.hash,
        # filters that are set define the plot data points
        kde.rtdc_ds.filter.all,
        # custom computation arguments influence e.g. Young's modulus
        kde.rtdc_ds.config.get("calculation", ""),
    ]


class ContourSpacingTooLarge(UserWarning):
    pass


class KernelDensityEstimator:
    def __init__(self, rtdc_ds):
        self.rtdc_ds = rtdc_ds

    @staticmethod
    def apply_scale(a, scale, feat):
        """Helper function for transforming an array to log-scale

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
                    warnings.warn(
                        f"Invalid values encounterd in np.log "
                        f"while scaling feature '{feat}'!"
                    )
        else:
            raise ValueError(
                f"`scale` must be either 'linear' or 'log', got '{scale}'!"
            )
        return b

    @staticmethod
    def check_feat_kde_applicability(xax: str, yax: str) -> bool:
        """Return True when it makes sense to compute KDE data"""
        if xax == yax:
            # Same feature -> just a line
            return False
        elif (xax in feat_const.FEATURES_MONOTONOUS
              and yax in feat_const.FEATURES_MONOTONOUS):
            # Monotonous features -> just a line
            return False
        else:
            return True

    @staticmethod
    def estimate_spacing(
        a, method,
        scale="linear", method_kw=None, feat="undefined", ret_scaled=False
    ):
        """Helper function for guessing the spacing/accuracy for KDE plots

        Parameters
        ----------
        a: ndarray
            feature data
        scale: str
            how the data should be scaled ("log" or "linear")
        method: callable
            Binning method to use
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
        asc = KernelDensityEstimator.apply_scale(a, scale, feat)
        # Apply multiplicator
        acc = method(asc, **method_kw)
        if ret_scaled:
            return acc, asc
        else:
            return acc

    @umbrella_cache(
        topic="kde-at",
        custom_handlers={"KernelDensityEstimator": kde_handler}
    )
    def get_at(
        self,
        xax="area_um",
        yax="deform",
        positions=None,
        kde_type="histogram",
        kde_kwargs=None,
        xscale="linear",
        yscale="linear",
        xacc=None,
        yacc=None,
    ):
        """Fast evaluation of the KDE via linear interpolation

        The KDE is computed via linear interpolation from the output
        of :meth:`KernelDensityEstimator.get_raster`.

        Parameters
        ----------
        xax, yax: str
            Identifier for X- and Y-axis (e.g. "area_um", "aspect", "deform")
        positions: list of two 1d ndarrays or ndarray of shape (2, N)
            The positions where the KDE will be computed. Note that
            the KDE estimate is computed from the points that
            are set in `self.rtdc_ds.filter.all`.
        kde_type: str
            The KDE method to use, see :const:`.kde_methods.methods`
        kde_kwargs: dict
            Additional keyword arguments to the KDE method
        xscale, yscale: str
            If set to "log", take the logarithm of the axes values before
            computing the KDE. This is useful when data are
            displayed on a log-scale. Defaults to "linear".
        xacc, yacc: float
            KDE grid spacing in x and y direction
            if set to None, will use :func:`.bin_width_percentile`

        Returns
        -------
        density : 1d ndarray
            The kernel density evaluated for the filtered events.
        """
        if kde_kwargs is None:
            kde_kwargs = {}
        xax = xax.lower()
        yax = yax.lower()
        kde_type = kde_type.lower()
        if kde_type not in methods:
            raise ValueError(f"Not a valid kde type: {kde_type}!")

        # Get data
        x = self.rtdc_ds[xax][self.rtdc_ds.filter.all]
        y = self.rtdc_ds[yax][self.rtdc_ds.filter.all]

        # Apply scale (no change for linear scale)
        if positions:
            xs = KernelDensityEstimator.apply_scale(positions[0], xscale, xax)
            ys = KernelDensityEstimator.apply_scale(positions[1], yscale, yax)
        else:
            xs = KernelDensityEstimator.apply_scale(x, xscale, xax)
            ys = KernelDensityEstimator.apply_scale(y, yscale, yax)

        if len(xs):
            xr, yr, density_grid = self.get_raster(
                xax=xax,
                yax=yax,
                kde_type=kde_type,
                kde_kwargs=kde_kwargs,
                xscale=xscale,
                yscale=yscale,
                xacc=xacc,
                yacc=yacc,
            )

            if xr.size:
                # Apply scale (no change for linear scale)
                xrs = KernelDensityEstimator.apply_scale(xr, xscale, xax)
                yrs = KernelDensityEstimator.apply_scale(yr, yscale, yax)

                # 'scipy.interp2d' has been removed in SciPy 1.14.0
                # https://scipy.github.io/devdocs/tutorial/interpolate/interp_transition_guide.html
                interp_func = RGI(
                    (xrs[:, 0], yrs[0, :]),
                    density_grid,
                    method="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                density = interp_func((xs, ys))
            else:
                # We don't have a raster to interpolate on. This can happen
                # when a feature has all-equal values.
                density = np.full(len(xs), np.nan)
        else:
            # No input coordinates.
            density = np.array([])

        return density

    @umbrella_cache(
        topic="kde-contour-lines",
        custom_handlers={"KernelDensityEstimator": kde_handler},
    )
    def get_contour_lines(
        self,
        quantiles=None,
        xax="area_um",
        yax="deform",
        kde_type="histogram",
        kde_kwargs=None,
        xscale="linear",
        yscale="linear",
        xacc=None,
        yacc=None,
        ret_levels=False,
    ):
        """Compute contour lines for a given kernel density estimate.

        Parameters
        ----------
        quantiles: list or array of floats
            KDE Quantiles for which contour levels are computed. The
            values must be between 0 and 1. If set to None, use
            [0.5, 0.95] as default.
        xax, yax: str
            Identifier for X- and Y-axis (e.g. "area_um", "aspect", "deform")
        kde_type: str
            The KDE method to use
        kde_kwargs: dict
            Additional keyword arguments to the KDE method
        xscale, yscale: str
            If set to "log", take the logarithm of the axis-values before
            computing the KDE. This is useful when data are
            displayed on a log-scale. Defaults to "linear".
        xacc, yacc: float
            KDE grid spacing in x and y direction; Affects contour shape.
            If set to None, will use :func:`.find_smooth_contour_spacing`
        ret_levels: bool
            If set to True, return the levels of the contours
            (default: False)

        Returns
        -------
        contour_lines: list of lists (of lists)
            For every number in `quantiles`, this list contains a list of
            corresponding contour lines. Each contour line is a 2D
            array of shape (N, 2), where N is the number of points in the
            contour line.
        levels: list of floats
            The density levels corresponding to each number in `quantiles`.
            Only returned if `ret_levels` is set to True.
        """
        if not quantiles:
            quantiles = [0.5, 0.95]

        if not KernelDensityEstimator.check_feat_kde_applicability(xax, yax):
            # There is no point in computing contours.
            contours = [[]] * len(quantiles)
            levels = [np.nan] * len(quantiles)
            if ret_levels:
                return contours, levels
            else:
                return contours

        if xacc is None or yacc is None:
            # workaround for circular import
            from .smooth_contour import find_smooth_contour_spacing
            sc_res = find_smooth_contour_spacing(
                [self.rtdc_ds],
                xax=xax,
                yax=yax,
                xrange=(np.min(self.rtdc_ds[xax]), np.max(self.rtdc_ds[xax])),
                yrange=(np.min(self.rtdc_ds[yax]), np.max(self.rtdc_ds[yax])),
                quantiles=quantiles,
                xscale=xscale,
                yscale=yscale,
            )
            xacc = xacc or sc_res["spacing x"]
            yacc = yacc or sc_res["spacing y"]

        try:
            x, y, density = self.get_raster(
                xax=xax,
                yax=yax,
                xacc=xacc,
                yacc=yacc,
                xscale=xscale,
                yscale=yscale,
                kde_type=kde_type,
                kde_kwargs=kde_kwargs,
            )
        except ValueError:
            # most-likely there is nothing to compute a contour for
            return []
        if density.shape[0] < 3 or density.shape[1] < 3:
            warnings.warn(
                "Contour not possible; spacing may be too large!",
                ContourSpacingTooLarge,
            )
            return []
        levels = get_quantile_levels(
            density=density,
            x=x,
            y=y,
            xp=self.rtdc_ds[xax][self.rtdc_ds.filter.all],
            yp=self.rtdc_ds[yax][self.rtdc_ds.filter.all],
            q=np.array(quantiles),
            normalize=False,
        )
        contours = []
        # Normalize levels to [0, 1]
        nlevels = np.array(levels) / density.max()
        for nlev in nlevels:
            # make sure that the contour levels are not at the boundaries
            if not (
                np.allclose(nlev, 0, atol=1e-12, rtol=0)
                or np.allclose(nlev, 1, atol=1e-12, rtol=0)
            ):
                cc = find_contours_level(density, x=x, y=y, level=nlev)
                contours.append(cc)
            else:
                contours.append([])
        if ret_levels:
            return contours, levels
        else:
            return contours

    @umbrella_cache(
        topic="kde-raster",
        custom_handlers={"KernelDensityEstimator": kde_handler}
    )
    def get_raster(
        self,
        xax="area_um",
        yax="deform",
        kde_type="histogram",
        kde_kwargs=None,
        xscale="linear",
        yscale="linear",
        xacc=None,
        yacc=None,
    ):
        """Evaluate the kernel density estimate on a grid

        Parameters
        ----------
        xax, yax: str
            Identifier for X- and Y-axis (e.g. "area_um", "aspect", "deform")
        kde_type: str
            The KDE method to use
        kde_kwargs: dict
            Additional keyword arguments to the KDE method
        xscale, yscale: str
            If set to "log", take the logarithm of the axes values before
            computing the KDE. This is useful when data are
            displayed on a log-scale. Defaults to "linear".
        xacc, yacc: float
            Grid spacing size for the axes;
            If set to None, will use :func:`bin_width_percentile`

        Returns
        -------
        X, Y, Z : coordinates
            The kernel density Z evaluated on a rectangular grid (X,Y).
        """
        if kde_kwargs is None:
            kde_kwargs = {}

        kde_type = kde_type.lower()
        if kde_type not in methods:
            raise ValueError(f"Not a valid kde type: {kde_type}!")

        # Get data
        x = self.rtdc_ds[xax][self.rtdc_ds.filter.all]
        y = self.rtdc_ds[yax][self.rtdc_ds.filter.all]

        xs = KernelDensityEstimator.apply_scale(x, scale=xscale, feat=xax)
        ys = KernelDensityEstimator.apply_scale(y, scale=yscale, feat=yax)

        if xacc is None or xacc == 0:
            xacc = bin_width_percentile(xs)

        if yacc is None or yacc == 0:
            yacc = bin_width_percentile(ys)

        # Ignore infs and nans
        bad = get_bad_vals(xs, ys)
        xc = xs[~bad]
        yc = ys[~bad]

        if xc.size:
            # Compute the mesh for rastering
            xnum = int(np.ceil((xc.max() - xc.min()) / xacc))
            ynum = int(np.ceil((yc.max() - yc.min()) / yacc))

            xlin = np.linspace(xc.min(), xc.max(), xnum, endpoint=True)
            ylin = np.linspace(yc.min(), yc.max(), ynum, endpoint=True)

            xmesh, ymesh = np.meshgrid(xlin, ylin, indexing="ij")

            # Compute the KDE for each point on the mesh
            kde_fct = methods[kde_type]
            density = kde_fct(
                events_x=xs, events_y=ys, xout=xmesh, yout=ymesh, **kde_kwargs
            )
        else:
            xmesh, ymesh = np.meshgrid([0, 1], [0, 1], indexing="ij")
            density = np.array(np.nan * xmesh)

        # Convert mesh back to linear scale if applicable
        if xscale == "log":
            xmesh = np.exp(xmesh)
        if yscale == "log":
            ymesh = np.exp(ymesh)

        return xmesh, ymesh, density

    @umbrella_cache(
        topic="kde-scatter",
        custom_handlers={"KernelDensityEstimator": kde_handler}
    )
    def get_scatter(
        self,
        xax="area_um",
        yax="deform",
        positions=None,
        kde_type="histogram",
        kde_kwargs=None,
        xscale="linear",
        yscale="linear",
    ):
        """Evaluate the KDE at specific positions

        The KDE is evaluated with the `kde_type` function for every point,
        which is significantly slower than using interpolation
        via :meth:`.KernelDensityEstimator.get_at`.

        Parameters
        ----------
        xax, yax: str
            Identifier for X- and Y-axis (e.g. "area_um", "aspect", "deform")
        positions: list of two 1d ndarrays or ndarray of shape (2, N)
            The positions where the KDE will be computed. Note that
            the KDE estimate is computed from the points that
            are set in `self.rtdc_ds.filter.all`.
        kde_type: str
            The KDE method to use, see :const:`.kde_methods.methods`
        kde_kwargs: dict
            Additional keyword arguments to the KDE method
        xscale, yscale: str
            If set to "log", take the logarithm of the axis-values before
            computing the KDE. This is useful when data are
            displayed on a log-scale. Defaults to "linear".

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
        if kde_type not in methods:
            raise ValueError(f"Not a valid kde type: {kde_type}!")

        # Get data
        x = self.rtdc_ds[xax][self.rtdc_ds.filter.all]
        y = self.rtdc_ds[yax][self.rtdc_ds.filter.all]

        # Apply scale (no change for linear scale)
        xs = KernelDensityEstimator.apply_scale(x, xscale, xax)
        ys = KernelDensityEstimator.apply_scale(y, yscale, yax)

        if positions is None:
            px = None
            py = None
        else:
            px = KernelDensityEstimator.apply_scale(positions[0], xscale, xax)
            py = KernelDensityEstimator.apply_scale(positions[1], yscale, yax)

        kde_fct = methods[kde_type]
        if len(x):
            density = kde_fct(
                events_x=xs, events_y=ys, xout=px, yout=py, **kde_kwargs
            )
        else:
            density = np.array([])

        return density
