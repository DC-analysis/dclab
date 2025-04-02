import warnings

import numpy as np

from .methods import bin_width_doane, get_bad_vals, methods


class KernelDensityEstimator:
    def __init__(self, rtdc_ds):
        self.rtdc_ds = rtdc_ds

    @staticmethod
    def apply_scale(a, scale, feat):
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
                    warnings.warn(f"Invalid values encounterd in np.log "
                                  f"while scaling feature '{feat}'!")
        else:
            raise ValueError(f"`scale` must be either 'linear' or 'log', "
                             f"got '{scale}'!")
        return b

    @staticmethod
    def get_spacing(a, method, scale="linear", method_kw=None,
                    feat="undefined", ret_scaled=False):
        """Convenience function for computing the contour spacing

        Parameters
        ----------
        a: ndarray
            feature data
        scale: str
            how the data should be scaled ("log" or "linear")
        method: callable
            KDE spacing method to use
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

    def get_contour(self, xax="area_um", yax="deform", xacc=None, yacc=None,
                    kde_type="histogram", kde_kwargs=None, xscale="linear",
                    yscale="linear"):
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
        if kde_type not in methods:
            raise ValueError(f"Not a valid kde type: {kde_type}!")

        # Get data
        x = self.rtdc_ds[xax][self.rtdc_ds.filter.all]
        y = self.rtdc_ds[yax][self.rtdc_ds.filter.all]

        xacc_sc, xs = self.get_spacing(
            a=x,
            feat=xax,
            scale=xscale,
            method=bin_width_doane,
            ret_scaled=True)

        yacc_sc, ys = self.get_spacing(
            a=y,
            feat=yax,
            scale=yscale,
            method=bin_width_doane,
            ret_scaled=True)

        if xacc is None or xacc == 0:
            xacc = xacc_sc / 5

        if yacc is None or yacc == 0:
            yacc = yacc_sc / 5

        # Ignore infs and nans
        bad = get_bad_vals(xs, ys)
        xc = xs[~bad]
        yc = ys[~bad]

        xnum = int(np.ceil((xc.max() - xc.min()) / xacc))
        ynum = int(np.ceil((yc.max() - yc.min()) / yacc))

        xlin = np.linspace(xc.min(), xc.max(), xnum, endpoint=True)
        ylin = np.linspace(yc.min(), yc.max(), ynum, endpoint=True)

        xmesh, ymesh = np.meshgrid(xlin, ylin, indexing="ij")

        kde_fct = methods[kde_type]
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

    def get_scatter(self, xax="area_um", yax="deform", positions=None,
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
            are set in `self.rtdc_ds.filter.all`.
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
        if kde_type not in methods:
            raise ValueError(f"Not a valid kde type: {kde_type}!")

        # Get data
        x = self.rtdc_ds[xax][self.rtdc_ds.filter.all]
        y = self.rtdc_ds[yax][self.rtdc_ds.filter.all]

        # Apply scale (no change for linear scale)
        xs = self.apply_scale(x, xscale, xax)
        ys = self.apply_scale(y, yscale, yax)

        if positions is None:
            posx = None
            posy = None
        else:
            posx = self.apply_scale(positions[0], xscale, xax)
            posy = self.apply_scale(positions[1], yscale, yax)

        kde_fct = methods[kde_type]
        if len(x):
            density = kde_fct(events_x=xs, events_y=ys,
                              xout=posx, yout=posy,
                              **kde_kwargs)
        else:
            density = np.array([])

        return density
