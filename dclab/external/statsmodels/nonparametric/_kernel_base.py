"""
Module containing the base object for multivariate kernel density and
regression, plus some utilities.
"""
import numpy as np

from . import kernels


kernel_func = dict(gaussian=kernels.gaussian)
has_joblib = False


class GenericKDE(object):
    """
    Base class for density estimation and regression KDE classes.
    """

    def _compute_bw(self, bw):
        """
        Computes the bandwidth of the data.

        Parameters
        ----------
        bw: array_like or str
            If array_like: user-specified bandwidth.
            If a string, should be one of:

                - cv_ml: cross validation maximum likelihood
                - normal_reference: normal reference rule of thumb
                - cv_ls: cross validation least squares

        Notes
        -----
        The default values for bw is 'normal_reference'.
        """
        if bw is None:
            bw = 'normal_reference'

        if not isinstance(bw, str):
            self._bw_method = "user-specified"
            res = np.asarray(bw)
        else:
            # The user specified a bandwidth selection method
            self._bw_method = bw
            # Workaround to avoid instance methods in __dict__
            if bw == 'normal_reference':
                bwfunc = self._normal_reference
            elif bw == 'cv_ml':
                bwfunc = self._cv_ml
            else:  # bw == 'cv_ls'
                bwfunc = self._cv_ls
            res = bwfunc()

        return res

    def _set_defaults(self, defaults):
        """Sets the default values for the efficient estimation"""
        self.n_res = defaults.n_res
        self.n_sub = defaults.n_sub
        self.randomize = defaults.randomize
        self.return_median = defaults.return_median
        self.efficient = defaults.efficient
        self.return_only_bw = defaults.return_only_bw
        self.n_jobs = defaults.n_jobs


class EstimatorSettings(object):
    """
    Object to specify settings for density estimation or regression.

    `EstimatorSettings` has several proporties related to how bandwidth
    estimation for the `KDEMultivariate`, `KDEMultivariateConditional`,
    `KernelReg` and `CensoredKernelReg` classes behaves.

    Parameters
    ----------
    efficient: bool, optional
        If True, the bandwidth estimation is to be performed
        efficiently -- by taking smaller sub-samples and estimating
        the scaling factor of each subsample.  This is useful for large
        samples (nobs >> 300) and/or multiple variables (k_vars > 3).
        If False (default), all data is used at the same time.
    randomize: bool, optional
        If True, the bandwidth estimation is to be performed by
        taking `n_res` random resamples (with replacement) of size `n_sub` from
        the full sample.  If set to False (default), the estimation is
        performed by slicing the full sample in sub-samples of size `n_sub` so
        that all samples are used once.
    n_sub: int, optional
        Size of the sub-samples.  Default is 50.
    n_res: int, optional
        The number of random re-samples used to estimate the bandwidth.
        Only has an effect if ``randomize == True``.  Default value is 25.
    return_median: bool, optional
        If True (default), the estimator uses the median of all scaling factors
        for each sub-sample to estimate the bandwidth of the full sample.
        If False, the estimator uses the mean.
    return_only_bw: bool, optional
        If True, the estimator is to use the bandwidth and not the
        scaling factor.  This is *not* theoretically justified.
        Should be used only for experimenting.
    n_jobs : int, optional
        The number of jobs to use for parallel estimation with
        ``joblib.Parallel``.  Default is -1, meaning ``n_cores - 1``, with
        ``n_cores`` the number of available CPU cores.
        See the `joblib documentation
        <https://pythonhosted.org/joblib/parallel.html>`_ for more details.

    Examples
    --------
    >>> settings = EstimatorSettings(randomize=True, n_jobs=3)
    >>> k_dens = KDEMultivariate(data, var_type, defaults=settings)

    """

    def __init__(self, efficient=False, randomize=False, n_res=25, n_sub=50,
                 return_median=True, return_only_bw=False, n_jobs=-1):
        self.efficient = efficient
        self.randomize = randomize
        self.n_res = n_res
        self.n_sub = n_sub
        self.return_median = return_median
        self.return_only_bw = return_only_bw  # TODO: remove this?
        self.n_jobs = n_jobs


def _adjust_shape(dat, k_vars):
    """ Returns an array of shape (nobs, k_vars) for use with `gpke`."""
    dat = np.asarray(dat)
    if dat.ndim > 2:
        dat = np.squeeze(dat)
    if dat.ndim == 1 and k_vars > 1:  # one obs many vars
        nobs = 1
    elif dat.ndim == 1 and k_vars == 1:  # one obs one var
        nobs = len(dat)
    else:
        if np.shape(dat)[0] == k_vars and np.shape(dat)[1] != k_vars:
            dat = dat.T

        nobs = np.shape(dat)[0]  # ndim >1 so many obs many vars

    dat = np.reshape(dat, (nobs, k_vars))
    return dat


def gpke(bw, data, data_predict, var_type, ckertype='gaussian',
         okertype='wangryzin', ukertype='aitchisonaitken', tosum=True):
    r"""
    Returns the non-normalized Generalized Product Kernel Estimator

    Parameters
    ----------
    bw: 1-D ndarray
        The user-specified bandwidth parameters.
    data: 1D or 2-D ndarray
        The training data.
    data_predict: 1-D ndarray
        The evaluation points at which the kernel estimation is performed.
    var_type: str, optional
        The variable type (continuous, ordered, unordered).
    ckertype: str, optional
        The kernel used for the continuous variables.
    okertype: str, optional
        The kernel used for the ordered discrete variables.
    ukertype: str, optional
        The kernel used for the unordered discrete variables.
    tosum : bool, optional
        Whether or not to sum the calculated array of densities.  Default is
        True.

    Returns
    -------
    dens: array-like
        The generalized product kernel density estimator.

    Notes
    -----
    The formula for the multivariate kernel estimator for the pdf is:

    .. math:: f(x)=\frac{1}{nh_{1}...h_{q}}\sum_{i=1}^
                        {n}K\left(\frac{X_{i}-x}{h}\right)

    where

    .. math:: K\left(\frac{X_{i}-x}{h}\right) =
                k\left( \frac{X_{i1}-x_{1}}{h_{1}}\right)\times
                k\left( \frac{X_{i2}-x_{2}}{h_{2}}\right)\times...\times
                k\left(\frac{X_{iq}-x_{q}}{h_{q}}\right)
    """
    kertypes = dict(c=ckertype, o=okertype, u=ukertype)

    Kval = np.empty(data.shape)
    for ii, vtype in enumerate(var_type):
        func = kernel_func[kertypes[vtype]]
        Kval[:, ii] = func(bw[ii], data[:, ii], data_predict[ii])

    iscontinuous = np.array([c == 'c' for c in var_type])
    dens = Kval.prod(axis=1) / np.prod(bw[iscontinuous])
    if tosum:
        return dens.sum(axis=0)
    else:
        return dens
