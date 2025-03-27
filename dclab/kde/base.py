import warnings

import numpy as np
import scipy.interpolate as spint
from scipy.interpolate import RectBivariateSpline
from scipy.stats import gaussian_kde, skew

from ..cached import Cache
from ..external.skimage.measure import find_contours, points_in_poly
from ..external.statsmodels.nonparametric.kernel_density import KDEMultivariate


class KernelDensityEtimator:
    def __init__(self, rtdc_ds):
        self.rtdc_ds = rtdc_ds

        self.methods = {
            "gauss": self.kde_gauss,
            "histogram": self.kde_histogram,
            "none": self.kde_none,
            "multivariate": self.kde_multivariate
        }

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
                    warnings.warn(f"Invalid values encounterd in np.log "
                                  f"while scaling feature '{feat}'!")
        else:
            raise ValueError(f"`scale` must be either 'linear' or 'log', "
                             f"got '{scale}'!")
        return b

    @staticmethod
    def get_bad_vals(x, y):
        return np.isnan(x) | np.isinf(x) | np.isnan(y) | np.isinf(y)

    @staticmethod
    def ignore_nan_inf(kde_method):
        """Ignores nans and infs from the input data

        Invalid positions in the resulting density are set to nan.
        """
        def new_kde_method(self, events_x, events_y, xout=None, yout=None,
                           *args, **kwargs):
            bad_in = self.get_bad_vals(events_x, events_y)
            if xout is None:
                density = np.zeros_like(events_x, dtype=np.float64)
                bad_out = bad_in
                xo = yo = None
            else:
                density = np.zeros_like(xout, dtype=np.float64)
                bad_out = self.get_bad_vals(xout, yout)
                xo = xout[~bad_out]
                yo = yout[~bad_out]
            # Filter events
            ev_x = events_x[~bad_in]
            ev_y = events_y[~bad_in]
            density[~bad_out] = kde_method(self, ev_x, ev_y,
                                           xo, yo,
                                           *args, **kwargs)
            density[bad_out] = np.nan
            return density

        doc_add = (
            "\n    Notes\n"
            "    -----\n"
            "    This is a wrapped version that ignores nan and inf values."
            )
        new_kde_method.__doc__ = kde_method.__doc__ + doc_add

        return new_kde_method

    def get_spacing(self, a, method, scale="linear", method_kw=None,
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
        asc = self._apply_scale(a, scale, feat)
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
        if kde_type not in self.methods:
            raise ValueError(f"Not a valid kde type: {kde_type}!")

        # Get data
        x = self.rtdc_ds[xax][self.rtdc_ds.filter.all]
        y = self.rtdc_ds[yax][self.rtdc_ds.filter.all]

        xacc_sc, xs = self.get_spacing(
            a=x,
            feat=xax,
            scale=xscale,
            method=self.bin_width_doane,
            ret_scaled=True)

        yacc_sc, ys = self.get_spacing(
            a=y,
            feat=yax,
            scale=yscale,
            method=self.bin_width_doane,
            ret_scaled=True)

        if xacc is None or xacc == 0:
            xacc = xacc_sc / 5

        if yacc is None or yacc == 0:
            yacc = yacc_sc / 5

        # Ignore infs and nans
        bad = self.get_bad_vals(xs, ys)
        xc = xs[~bad]
        yc = ys[~bad]

        xnum = int(np.ceil((xc.max() - xc.min()) / xacc))
        ynum = int(np.ceil((yc.max() - yc.min()) / yacc))

        xlin = np.linspace(xc.min(), xc.max(), xnum, endpoint=True)
        ylin = np.linspace(yc.min(), yc.max(), ynum, endpoint=True)

        xmesh, ymesh = np.meshgrid(xlin, ylin, indexing="ij")

        kde_fct = self.methods[kde_type]
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
        if kde_type not in self.methods:
            raise ValueError(f"Not a valid kde type: {kde_type}!")

        # Get data
        x = self.rtdc_ds[xax][self.rtdc_ds.filter.all]
        y = self.rtdc_ds[yax][self.rtdc_ds.filter.all]

        # Apply scale (no change for linear scale)
        xs = self._apply_scale(x, xscale, xax)
        ys = self._apply_scale(y, yscale, yax)

        if positions is None:
            posx = None
            posy = None
        else:
            posx = self._apply_scale(positions[0], xscale, xax)
            posy = self._apply_scale(positions[1], yscale, yax)

        kde_fct = self.methods[kde_type]
        if len(x):
            density = kde_fct(events_x=xs, events_y=ys,
                              xout=posx, yout=posy,
                              **kde_kwargs)
        else:
            density = np.array([])

        return density

    def find_contours_level(self, density, x, y, level, closed=False):
        """Find iso-valued density contours for a given level value

        Parameters
        ----------
        density: 2d ndarray of shape (M, N)
            Kernel density estimate (KDE) for which to compute the contours
        x: 2d ndarray of shape (M, N) or 1d ndarray of size M
            X-values corresponding to `density`
        y: 2d ndarray of shape (M, N) or 1d ndarray of size M
            Y-values corresponding to `density`
        level: float between 0 and 1
            Value along which to find contours in `density` relative
            to its maximum
        closed: bool
            Whether to close contours at the KDE support boundaries

        Returns
        -------
        contours: list of ndarrays of shape (P, 2)
            Contours found for the given level value

        See Also
        --------
        skimage.measure.find_contours: Contour finding algorithm used
        """
        if level >= 1 or level <= 0:
            raise ValueError(f"`level` must be in (0,1), got '{level}'!")
        # level relative to maximum
        level = level * density.max()
        # xy coordinates
        if len(x.shape) == 2:
            assert np.all(x[:, 0] == x[:, 1])
            x = x[:, 0]
        if len(y.shape) == 2:
            assert np.all(y[0, :] == y[1, :])
            y = y[0, :]
        if closed:
            # find closed contours
            density = np.pad(density, ((1, 1), (1, 1)), mode="constant")
            offset = 1
        else:
            # leave contours open at kde boundary
            offset = 0

        conts_idx = find_contours(density, level)
        conts_xy = []

        for cc in conts_idx:
            cx = np.interp(x=cc[:, 0]-offset,
                           xp=range(x.size),
                           fp=x)
            cy = np.interp(x=cc[:, 1]-offset,
                           xp=range(y.size),
                           fp=y)
            conts_xy.append(np.stack((cx, cy), axis=1))

        return conts_xy

    def get_quantile_levels(self, density, x, y, xp, yp, q, normalize=True):
        """Compute density levels for given quantiles by interpolation

        For a given 2D density, compute the density levels at which
        the resulting contours contain the fraction `1-q` of all
        data points. E.g. for a measurement of 1000 events, all
        contours at the level corresponding to a quantile of
        `q=0.95` (95th percentile) contain 50 events (5%).

        Parameters
        ----------
        density: 2d ndarray of shape (M, N)
            Kernel density estimate for which to compute the contours
        x: 2d ndarray of shape (M, N) or 1d ndarray of size M
            X-values corresponding to `density`
        y: 2d ndarray of shape (M, N) or 1d ndarray of size M
            Y-values corresponding to `density`
        xp: 1d ndarray of size D
            Event x-data from which to compute the quantile
        yp: 1d ndarray of size D
            Event y-data from which to compute the quantile
        q: array_like or float between 0 and 1
            Quantile along which to find contours in `density` relative
            to its maximum
        normalize: bool
            Whether output levels should be normalized to the maximum
            of `density`

        Returns
        -------
        level: np.ndarray or float
            Contours level(s) corresponding to the given quantile

        Notes
        -----
        NaN-values events in `xp` and `yp` are ignored.
        """
        # xy coordinates
        if len(x.shape) == 2:
            assert np.all(x[:, 0] == x[:, 1])
            x = x[:, 0]
        if len(y.shape) == 2:
            assert np.all(y[0, :] == y[1, :])
            y = y[0, :]

        # remove bad events
        bad = self.get_bad_vals(xp, yp)
        xp = xp[~bad]
        yp = yp[~bad]

        # Normalize interpolation data such that the spacing for
        # x and y is about the same during interpolation.
        x_norm = x.max()
        x = x / x_norm
        xp = xp / x_norm

        y_norm = y.max()
        y = y / y_norm
        yp = yp / y_norm

        # Perform interpolation
        dp = spint.interpn((x, y), density,
                           (xp, yp),
                           method='linear',
                           bounds_error=False,
                           fill_value=0)

        if normalize:
            dp /= density.max()

        if not np.isscalar(q):
            q = np.array(q)
        plev = np.nanpercentile(dp, q=q*100)
        return plev

    def _find_quantile_level(self, density, x, y, xp, yp, quantile, acc=.01,
                             ret_err=False):
        """Find density level for a given data quantile by iteration

        Parameters
        ----------
        density: 2d ndarray of shape (M, N)
            Kernel density estimate for which to compute the contours
        x: 2d ndarray of shape (M, N) or 1d ndarray of size M
            X-values corresponding to `density`
        y: 2d ndarray of shape (M, N) or 1d ndarray of size M
            Y-values corresponding to `density`
        xp: 1d ndarray of size D
            Event x-data from which to compute the quantile
        yp: 1d ndarray of size D
            Event y-data from which to compute the quantile
        quantile: float between 0 and 1
            Quantile along which to find contours in `density` relative
            to its maximum
        acc: float
            Desired absolute accuracy (stopping criterion) of the
            contours
        ret_err: bool
            If True, also return the absolute error

        Returns
        -------
        level: float
            Contours level corresponding to the given quantile

        Notes
        -----
        A much more faster method (using interpolation) is implemented in
        :func:`get_quantile_levels`.
        NaN-values events in `xp` and `yp` are ignored.

        See Also
        --------
        skimage.measure.find_contours: Contour finding algorithm
        """
        if quantile >= 1 or quantile <= 0:
            raise ValueError(f"Invalid value for `quantile`: {quantile}")

        # remove bad events
        bad = self.get_bad_vals(xp, yp)
        xp = xp[~bad]
        yp = yp[~bad]
        points = np.concatenate((xp.reshape(-1, 1), yp.reshape(-1, 1)), axis=1)

        # initial guess
        level = quantile
        # error of current iteration
        err = 1
        # iteration factor (guarantees convergence)
        itfac = 1
        # total number of events
        nev = xp.size

        while np.abs(err) > acc:
            # compute contours
            conts = self.find_contours_level(density, x, y, level, closed=True)
            # compute number of points in contour
            isin = 0
            pi = np.array(points, copy=True)
            for cc in conts:
                pinc = points_in_poly(points=pi, verts=cc)
                isin += np.sum(pinc)
                # ignore these points for the other contours
                pi = pi[~pinc]
            err = quantile - (nev - isin) / nev
            level += err * itfac
            itfac *= .9

        if ret_err:
            return level, err
        else:
            return level

    def bin_num_doane(self, a):
        """Compute number of bins based on Doane's formula

        Notes
        -----
        If the bin width cannot be determined, then a bin
        number of 5 is returned.

        See Also
        --------
        bin_width_doane: method used to compute the bin width
        """
        bad = np.isnan(a) | np.isinf(a)
        data = a[~bad]
        acc = self.bin_width_doane(a)
        if acc == 0 or np.isnan(acc):
            num = 5
        else:
            num = int(np.round((data.max() - data.min()) / acc))
        return num

    def bin_width_doane(self, a):
        """Compute contour spacing based on Doane's formula

        References
        ----------
        - `<https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width>`_
        - `<https://stats.stackexchange.com/questions/55134/
        doanes-formula-for-histogram-binning>`_

        Notes
        -----
        Doane's formula is actually designed for histograms. This
        function is kept here for backwards-compatibility reasons.
        It is highly recommended to use :func:`bin_width_percentile`
        instead.
        """
        bad = np.isnan(a) | np.isinf(a)
        data = a[~bad]
        n = data.size
        g1 = skew(data)
        sigma_g1 = np.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
        k = 1 + np.log2(n) + np.log2(1 + np.abs(g1) / sigma_g1)
        acc = (data.max() - data.min()) / k
        return acc

    def bin_width_percentile(self, a):
        """Compute contour spacing based on data percentiles

        The 10th and the 90th percentile of the input data are taken.
        The spacing then computes to the difference between those
        two percentiles divided by 23.

        Notes
        -----
        The Freedman–Diaconis rule uses the interquartile range and
        normalizes to the third root of len(a). Such things do not
        work very well for RT-DC data, because len(a) is huge. Here
        we use just the top and bottom 10th percentiles with a fixed
        normalization.
        """
        bad = np.isnan(a) | np.isinf(a)
        data = a[~bad]
        start = np.percentile(data, 10)
        end = np.percentile(data, 90)
        acc = (end - start) / 23
        return acc

    @ignore_nan_inf
    @Cache
    def kde_gauss(self, events_x, events_y, xout=None, yout=None):
        """ Gaussian Kernel Density Estimation

        Parameters
        ----------
        events_x, events_y: 1D ndarray
            The input points for kernel density estimation. Input
            is flattened automatically.
        xout, yout: ndarray
            The coordinates at which the KDE should be computed.
            If set to none, input coordinates are used.

        Returns
        -------
        density: ndarray, same shape as `xout`
            The KDE for the points in (xout, yout)

        See Also
        --------
        `scipy.stats.gaussian_kde`
        """
        valid_combi = ((xout is None and yout is None) or
                       (xout is not None and yout is not None)
                       )
        if not valid_combi:
            raise ValueError("Both `xout` and `yout` must be (un)set.")

        if xout is None and yout is None:
            xout = events_x
            yout = events_y

        try:
            estimator = gaussian_kde([events_x.flatten(), events_y.flatten()])
            density = estimator.evaluate([xout.flatten(), yout.flatten()])
        except np.linalg.LinAlgError:
            # LinAlgError occurs when matrix to solve is singular (issue #117)
            density = np.zeros(xout.shape)*np.nan
        return density.reshape(xout.shape)

    @ignore_nan_inf
    @Cache
    def kde_histogram(self, events_x, events_y, xout=None, yout=None,
                      bins=None):
        """ Histogram-based Kernel Density Estimation

        Parameters
        ----------
        events_x, events_y: 1D ndarray
            The input points for kernel density estimation. Input
            is flattened automatically.
        xout, yout: ndarray
            The coordinates at which the KDE should be computed.
            If set to none, input coordinates are used.
        bins: tuple (binsx, binsy)
            The number of bins to use for the histogram.

        Returns
        -------
        density: ndarray, same shape as `xout`
            The KDE for the points in (xout, yout)

        See Also
        --------
        `numpy.histogram2d`
        `scipy.interpolate.RectBivariateSpline`
        """
        valid_combi = ((xout is None and yout is None) or
                       (xout is not None and yout is not None)
                       )
        if not valid_combi:
            raise ValueError("Both `xout` and `yout` must be (un)set.")

        if xout is None and yout is None:
            xout = events_x
            yout = events_y

        if bins is None:
            bins = (max(5, self.bin_num_doane(events_x)),
                    max(5, self.bin_num_doane(events_y)))

        # Compute the histogram
        hist2d, xedges, yedges = np.histogram2d(x=events_x,
                                                y=events_y,
                                                bins=bins,
                                                density=True)
        xip = xedges[1:]-(xedges[1]-xedges[0])/2
        yip = yedges[1:]-(yedges[1]-yedges[0])/2

        estimator = RectBivariateSpline(x=xip, y=yip, z=hist2d)
        density = estimator.ev(xout, yout)
        density[density < 0] = 0

        return density.reshape(xout.shape)

    def kde_none(self, events_x, events_y, xout=None, yout=None):
        """No Kernel Density Estimation

        Parameters
        ----------
        events_x, events_y: 1D ndarray
            The input points for kernel density estimation. Input
            is flattened automatically.
        xout, yout: ndarray
            The coordinates at which the KDE should be computed.
            If set to none, input coordinates are used.

        Returns
        -------
        density: ndarray, same shape as `xout`
            The KDE for the points in (xout, yout)

        Notes
        -----
        This method is a convenience method that always returns ones in the
        shape that the other methods in this module produce.
        """
        valid_combi = ((xout is None and yout is None) or
                       (xout is not None and yout is not None)
                       )
        if not valid_combi:
            raise ValueError("Both `xout` and `yout` must be (un)set.")

        if xout is None and yout is None:
            xout = events_x
            _ = events_y

        return np.ones(xout.shape)

    @ignore_nan_inf
    @Cache
    def kde_multivariate(self, events_x, events_y, xout=None, yout=None,
                         bw=None):
        """ Multivariate Kernel Density Estimation

        Parameters
        ----------
        events_x, events_y: 1D ndarray
            The input points for kernel density estimation. Input
            is flattened automatically.
        bw: tuple (bwx, bwy) or None
            The bandwith for kernel density estimation.
        xout, yout: ndarray
            The coordinates at which the KDE should be computed.
            If set to none, input coordinates are used.

        Returns
        -------
        density: ndarray, same shape as `xout`
            The KDE for the points in (xout, yout)

        See Also
        --------
        `statsmodels.nonparametric.kernel_density.KDEMultivariate`
        """
        valid_combi = ((xout is None and yout is None) or
                       (xout is not None and yout is not None)
                       )
        if not valid_combi:
            raise ValueError("Both `xout` and `yout` must be (un)set.")

        if xout is None and yout is None:
            xout = events_x
            yout = events_y
        if bw is None:
            # divide by 2 to make it comparable to histogram KDE
            bw = (self.bin_width_doane(events_x) / 2,
                  self.bin_width_doane(events_y) / 2)

        positions = np.vstack([xout.flatten(), yout.flatten()])
        estimator_ly = KDEMultivariate(data=[events_x.flatten(),
                                             events_y.flatten()],
                                       var_type='cc', bw=bw)

        density = estimator_ly.pdf(positions)
        return density.reshape(xout.shape)
