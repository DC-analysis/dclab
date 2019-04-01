"""
Multivariate Conditional and Unconditional Kernel Density Estimation
with Mixed Data Types.

References
----------
[1] Racine, J., Li, Q. Nonparametric econometrics: theory and practice.
    Princeton University Press. (2007)
[2] Racine, Jeff. "Nonparametric Econometrics: A Primer," Foundation
    and Trends in Econometrics: Vol 3: No 1, pp1-88. (2008)
    http://dx.doi.org/10.1561/0800000009
[3] Racine, J., Li, Q. "Nonparametric Estimation of Distributions
    with Categorical and Continuous Data." Working Paper. (2000)
[4] Racine, J. Li, Q. "Kernel Estimation of Multivariate Conditional
    Distributions Annals of Economics and Finance 5, 211-235 (2004)
[5] Liu, R., Yang, L. "Kernel estimation of multivariate
    cumulative distribution function."
    Journal of Nonparametric Statistics (2008)
[6] Li, R., Ju, G. "Nonparametric Estimation of Multivariate CDF
    with Categorical and Continuous Data." Working Paper
[7] Li, Q., Racine, J. "Cross-validated local linear nonparametric
    regression" Statistica Sinica 14(2004), pp. 485-512
[8] Racine, J.: "Consistent Significance Testing for Nonparametric
        Regression" Journal of Business & Economics Statistics
[9] Racine, J., Hart, J., Li, Q., "Testing the Significance of
        Categorical Predictor Variables in Nonparametric Regression
        Models", 2006, Econometric Reviews 25, 523-544

"""
from __future__ import division

import numpy as np

from ._kernel_base import GenericKDE, EstimatorSettings, gpke, _adjust_shape


__all__ = ['KDEMultivariate', 'EstimatorSettings']


class KDEMultivariate(GenericKDE):
    """
    Multivariate kernel density estimator.

    This density estimator can handle univariate as well as multivariate data,
    including mixed continuous / ordered discrete / unordered discrete data.
    It also provides cross-validated bandwidth selection methods (least
    squares, maximum likelihood).

    Parameters
    ----------
    data: list of ndarrays or 2-D ndarray
        The training data for the Kernel Density Estimation, used to determine
        the bandwidth(s).  If a 2-D array, should be of shape
        (num_observations, num_variables).  If a list, each list element is a
        separate observation.
    var_type: str
        The type of the variables:

            - c : continuous
            - u : unordered (discrete)
            - o : ordered (discrete)

        The string should contain a type specifier for each variable, so for
        example ``var_type='ccuo'``.
    bw: array_like or str, optional
        If an array, it is a fixed user-specified bandwidth.  If a string,
        should be one of:

            - normal_reference: normal reference rule of thumb (default)
            - cv_ml: cross validation maximum likelihood
            - cv_ls: cross validation least squares

    defaults: EstimatorSettings instance, optional
        The default values for (efficient) bandwidth estimation.

    Attributes
    ----------
    bw: array_like
        The bandwidth parameters.

    See Also
    --------
    KDEMultivariateConditional

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> nobs = 300
    >>> np.random.seed(1234)  # Seed random generator
    >>> c1 = np.random.normal(size=(nobs,1))
    >>> c2 = np.random.normal(2, 1, size=(nobs,1))

    Estimate a bivariate distribution and display the bandwidth found:

    >>> dens_u = sm.nonparametric.KDEMultivariate(data=[c1,c2],
    ...     var_type='cc', bw='normal_reference')
    >>> dens_u.bw
    array([ 0.39967419,  0.38423292])
    """

    def __init__(self, data, var_type, bw=None, defaults=None):
        self.var_type = var_type
        self.k_vars = len(self.var_type)
        self.data = _adjust_shape(data, self.k_vars)
        self.data_type = var_type
        self.nobs, self.k_vars = np.shape(self.data)
        if self.nobs <= self.k_vars:
            raise ValueError("The number of observations must be larger "
                             "than the number of variables.")
        defaults = EstimatorSettings() if defaults is None else defaults
        self._set_defaults(defaults)
        if not self.efficient:
            self.bw = self._compute_bw(bw)
        else:
            self.bw = self._compute_efficient(bw)

    def __repr__(self):
        """Provide something sane to print."""
        rpr = "KDE instance\n"
        rpr += "Number of variables: k_vars = " + str(self.k_vars) + "\n"
        rpr += "Number of samples:   nobs = " + str(self.nobs) + "\n"
        rpr += "Variable types:      " + self.var_type + "\n"
        rpr += "BW selection method: " + self._bw_method + "\n"
        return rpr

    def pdf(self, data_predict=None):
        r"""
        Evaluate the probability density function.

        Parameters
        ----------
        data_predict: array_like, optional
            Points to evaluate at.  If unspecified, the training data is used.

        Returns
        -------
        pdf_est: array_like
            Probability density function evaluated at `data_predict`.

        Notes
        -----
        The probability density is given by the generalized product kernel
        estimator:

        .. math:: K_{h}(X_{i},X_{j}) =
            \prod_{s=1}^{q}h_{s}^{-1}k\left(\frac{X_{is}-X_{js}}{h_{s}}\right)
        """
        if data_predict is None:
            data_predict = self.data
        else:
            data_predict = _adjust_shape(data_predict, self.k_vars)

        pdf_est = []
        for i in range(np.shape(data_predict)[0]):
            pdf_est.append(gpke(self.bw, data=self.data,
                                data_predict=data_predict[i, :],
                                var_type=self.var_type) / self.nobs)

        pdf_est = np.squeeze(pdf_est)
        return pdf_est

    def _get_class_vars_type(self):
        """Helper method to be able to pass needed vars to _compute_subset."""
        class_type = 'KDEMultivariate'
        class_vars = (self.var_type, )
        return class_type, class_vars
