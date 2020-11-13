"""R lme4 wrapper"""
import numbers
import warnings

import numpy as np

from .. import definitions as dfn
from ..rtdc_dataset.core import RTDCBase

from .rlibs import rpy2
from . import rsetup


class Lme4InstallWarning(UserWarning):
    pass


class Rlme4(object):
    def __init__(self, model="lmer", feature="deform"):
        """Perform an R-lme4 analysis with RT-DC data

        Parameters
        ----------
        model: str
            One of:

            - "lmer": linear mixed model using lme4's ``lmer``
            - "glmer+loglink": generalized linear mixed model using
              lme4's ``glmer`` with an additional a log-link function
              via the ``family=Gamma(link='log'))`` keyword.
        feature: str
            Dclab feature for which to compute the model
        """
        #: modeling method to use (e.g. "lmer")
        self.model = None
        #: dclab feature for which to perform the analysis
        self.feature = None
        #: list of [RTDCBase, column, repetition, chip_region]
        self.data = []

        #: model function
        self.r_func_model = "feature ~ group + (1 + group | repetition)"
        #: null model function
        self.r_func_nullmodel = "feature ~ (1 + group | repetition)"

        self.set_options(model=model, feature=feature)

        # Make sure that lme4 is available
        if not rsetup.has_lme4():
            warnings.warn("Installing lme4, this may take a while!",
                          Lme4InstallWarning)
            rsetup.install_lme4()
        rsetup.import_lme4()

    def add_dataset(self, ds, group, repetition):
        """Add a dataset to the analysis list

        Parameters
        ----------
        ds: RTDCBase
            Dataset
        group: str
            The group the measurement belongs to ("control" or
            "treatment")
        repetition: int
            Repetition of the measurement

        Notes
        -----
        - For each repetition, there must be a "treatment" and a
          "control" ``group``.
        - If you would like to perform a differential feature analysis,
          then you need to pass at least a reservoir and a channel
          dataset (with same parameters for `group` and `repetition`).
        """
        assert group in ["treatment", "control"]
        assert isinstance(ds, RTDCBase)
        assert isinstance(repetition, numbers.Integral)

        region = ds.config["setup"]["chip region"]
        # make sure there are no doublets
        for ii, dd in enumerate(self.data):
            if dd[1] == group and dd[2] == repetition and dd[3] == region:
                raise ValueError("A dataset with group '{}', ".format(group)
                                 + "repetition '{}', and ".format(repetition)
                                 + "'{}' region has already ".format(region)
                                 + "been added (index {})!".format(ii))

        self.data.append([ds, group, repetition, region])

    def check_data(self):
        """Perform sanity checks on ``self.data``"""
        # Check that we have enough data
        if len(self.data) < 3:
            msg = "Linear mixed effects models require repeated " \
                  + "measurements. Please add more repetitions."
            raise ValueError(msg)

    def fit(self, model=None, feature=None):
        """Perform (generalized) linear mixed-effects model fit

        The response variable is modeled using two linear mixed effect
        models:

        - model :const:`Rlme4.r_func_model` (random intercept +
          random slope model)
        - the null model :const:`Rlme4.r_func_nullmodel` (without
          the fixed effect introduced by the "treatment" group).

        Both models are compared in R using "anova" (from the
        R-package "stats" :cite:`Everitt1992`) which performs a
        likelihood ratio test to obtain the p-Value for the
        significance of the fixed effect (treatment).

        If the input datasets contain data from the "reservoir"
        region, then the analysis is performed for the differential
        feature.

        Parameters
        ----------
        model: str (optional)
            One of:

            - "lmer": linear mixed model using lme4's ``lmer``
            - "glmer+loglink": generalized linear mixed model using
              lme4's ``glmer`` with an additional log-link function
              via ``family=Gamma(link='log'))`` :cite:`lme4`
        feature: str (optional)
            dclab feature for which to compute the model

        Returns
        -------
        results: dict
            Dictionary with the results of the fitting process:

            - "anova p-value": Anova likelyhood ratio test (significance)
            - "feature": name of the feature used for the analysis
              ``self.feature``
            - "fixed effects intercept": Mean of ``self.feature`` for all
              controls; In the case of the "glmer+loglink" model, the intercept
              is already backtransformed from log space.
            - "fixed effects treatment": The fixed effect size between the mean
              of the controls and the mean of the treatments relative to
              "fixed effects intercept"; In the case of the "glmer+loglink"
              model, the fixed effect is already backtransformed from log
              space.
            - "fixed effects repetitions": The effects (intercept and
              treatment) for each repetition. The first axis defines
              intercept/treatment; the second axis enumerates the repetitions;
              thus the shape is (2, number of repetitions) and
              ``np.mean(results["fixed effects repetitions"], axis=1)`` is
              equivalent to the tuple (``results["fixed effects intercept"]``,
              ``results["fixed effects treatment"]``) for the "lmer" model.
              This does not hold for the "glmer+loglink" model, because
              of the non-linear inverse transform back from log space.
            - "is differential": Boolean indicating whether or not
              the analysis was performed for the differential (bootstrapped
              and subtracted reservoir from channel data) feature
            - "model": model name used for the analysis ``self.model``
            - "model converged": boolean indicating whether the model
              converged
            - "r anova": Anova model (exposed from R)
            - "r model summary": Summary of the model (exposed from R)
            - "r model coefficients": Model coefficient table (exposed from R)
            - "r stderr": errors and warnings from R
            - "r stdout": standard output from R
        """
        self.set_options(model=model, feature=feature)
        self.check_data()

        # Assemble dataset
        if self.is_differential():
            # bootstrap and compute differential features using reservoir
            features, groups, repetitions = self.get_differential_dataset()
        else:
            # regular feature analysis
            features = []
            groups = []
            repetitions = []
            for dd in self.data:
                features.append(self.get_feature_data(dd[1], dd[2]))
                groups.append(dd[1])
                repetitions.append(dd[2])

        # Fire up R
        with rsetup.AutoRConsole() as ac:
            r = rpy2.robjects.r

            # Load lme4
            rpy2.robjects.packages.importr("lme4")

            # Concatenate huge arrays for R
            r_features = rpy2.robjects.FloatVector(np.concatenate(features))
            _groups = []
            _repets = []
            for ii in range(len(features)):
                _groups.append(np.repeat(groups[ii], len(features[ii])))
                _repets.append(np.repeat(repetitions[ii], len(features[ii])))
            r_groups = rpy2.robjects.StrVector(np.concatenate(_groups))
            r_repetitions = rpy2.robjects.IntVector(np.concatenate(_repets))

            # Register groups and repetitions
            rpy2.robjects.globalenv["feature"] = r_features
            rpy2.robjects.globalenv["group"] = r_groups
            rpy2.robjects.globalenv["repetition"] = r_repetitions

            # Create a dataframe which contains all the data
            r_data = r["data.frame"](r_features, r_groups, r_repetitions)

            # Random intercept and random slope model
            if self.model == 'glmer+loglink':
                r_model = r["glmer"](self.r_func_model, r_data,
                                     family=r["Gamma"](link='log'))
                r_nullmodel = r["glmer"](self.r_func_nullmodel, r_data,
                                         family=r["Gamma"](link='log'))
            else:  # lmer
                r_model = r["lmer"](self.r_func_model, r_data)
                r_nullmodel = r["lmer"](self.r_func_nullmodel, r_data)

            # Anova analysis (increase verbosity by making models global)
            # Using anova is a very conservative way of determining
            # p values.
            rpy2.robjects.globalenv["Model"] = r_model
            rpy2.robjects.globalenv["NullModel"] = r_nullmodel
            r_anova = r("Anova = anova(Model, NullModel)")
            try:
                pvalue = r_anova.rx2["Pr(>Chisq)"][1]
            except ValueError:  # rpy2 2.9.4
                pvalue = r_anova[7][1]
            r_model_summary = r["summary"](r_model)
            r_model_coefficients = r["coef"](r_model)
            try:
                fe_reps = np.array(r_model_coefficients.rx2["repetition"])
            except ValueError:  # rpy2 2.9.4
                fe_reps = np.concatenate((
                    np.array(r_model_coefficients[0][0]).reshape(1, -1),
                    np.array(r_model_coefficients[0][1]).reshape(1, -1)),
                    axis=0)

            r_effects = r["data.frame"](r["coef"](r_model_summary))
            try:
                fe_icept = r_effects.rx2["Estimate"][0]
                fe_treat = r_effects.rx2["Estimate"][1]
            except ValueError:  # rpy2 2.9.4
                fe_icept = r_effects[0][0]
                fe_treat = r_effects[0][1]
            if self.model == "glmer+loglink":
                # transform back from log
                fe_treat = np.exp(fe_icept + fe_treat) - np.exp(fe_icept)
                fe_icept = np.exp(fe_icept)
                fe_reps[:, 1] = np.exp(fe_reps[:, 0] + fe_reps[:, 1]) \
                    - np.exp(fe_reps[:, 0])
                fe_reps[:, 0] = np.exp(fe_reps[:, 0])

            # convergence
            try:
                lme4l = r_model_summary.rx2["optinfo"].rx2["conv"].rx2["lme4"]
            except ValueError:  # rpy2 2.9.4
                lme4l = r_model_summary[17][3][1]

            if lme4l and "code" in lme4l.names:
                try:
                    conv_code = lme4l.rx2["code"]
                except ValueError:  # rpy2 2.9.4
                    conv_code = lme4l[0]
            else:
                conv_code = 0

        ret_dict = {
            "anova p-value": pvalue,
            "feature": self.feature,
            "fixed effects intercept": fe_icept,
            "fixed effects treatment": fe_treat,  # aka "fixed effect"
            "fixed effects repetitions": fe_reps,
            "is differential": self.is_differential(),
            "model": self.model,
            "model converged": conv_code == 0,
            "r anova": r_anova,
            "r model summary": r_model_summary,
            "r model coefficients": r_model_coefficients,
            "r stderr": ac.get_warnerrors(),
            "r stdout": ac.get_prints(),
        }
        return ret_dict

    def get_differential_dataset(self):
        """Return the differential dataset for channel/reservoir data

        The most famous use case is differential deformation. The idea
        is that you cannot tell what the difference in deformation
        from channel to reservoir is, because you never measure the
        same object in the reservoir and the channel. You usually just
        have two distributions. Comparing distributions is possible
        via bootstrapping. And then, instead of running the lme4
        analysis with the channel deformation data, it is run with
        the differential deformation (subtraction of the bootstrapped
        deformation distributions for channel and reservoir).
        """
        features = []
        groups = []
        repetitions = []
        # compute differential features
        for grp in sorted(set([dd[1] for dd in self.data])):
            # repetitions per groups
            grp_rep = sorted(set([dd[2] for dd in self.data if dd[1] == grp]))
            for rep in grp_rep:
                feat_cha = self.get_feature_data(grp, rep, region="channel")
                feat_res = self.get_feature_data(grp, rep, region="reservoir")
                bs_cha, bs_res = bootstrapped_median_distributions(feat_cha,
                                                                   feat_res)
                # differential feature
                features.append(bs_cha - bs_res)
                groups.append(grp)
                repetitions.append(rep)
        return features, groups, repetitions

    def get_feature_data(self, group, repetition, region="channel"):
        """Return array containing feature data

        Parameters
        ----------
        group: str
            Measurement group ("control" or "treatment")
        repetition: int
            Measurement repetition
        region: str
            Either "channel" or "reservoir"

        Returns
        -------
        fdata: 1d ndarray
            Feature data (Nans and Infs removed)
        """
        assert group in ["control", "treatment"]
        assert isinstance(repetition, numbers.Integral)
        assert region in ["reservoir", "channel"]
        for dd in self.data:
            if dd[1] == group and dd[2] == repetition and dd[3] == region:
                ds = dd[0]
                break
        else:
            raise ValueError("Dataset for group '{}', repetition".format(group)
                             + " '{}', and region".format(repetition)
                             + " '{}' not found!".format(region))
        fdata = ds[self.feature][ds.filter.all]
        fdata_valid = fdata[~np.logical_or(np.isnan(fdata), np.isinf(fdata))]
        return fdata_valid

    def is_differential(self):
        """Return True if the differential feature is computed for analysis

        This effectively just checks the regions of the datasets
        and returns True if any one of the regions is "reservoir".

        See Also
        --------
        get_differential_features: for an explanation
        """
        for dd in self.data:
            if dd[3] == "reservoir":
                return True
        else:
            return False

    def set_options(self, model=None, feature=None):
        """Set analysis options"""
        if model is not None:
            assert model in ["lmer", "glmer+loglink"]
            self.model = model
        if feature is not None:
            assert dfn.scalar_feature_exists(feature)
            self.feature = feature


def bootstrapped_median_distributions(a, b, bs_iter=1000, rs=117):
    """Compute the bootstrapped distributions for two arrays.

    Parameters
    ----------
    a, b: 1d ndarray of length N
        Input data
    bs_iter: int
        Number of bootstrapping iterations to perform
        (outtput size).
    rs: int
        Random state seed for random number generator

    Returns
    -------
    median_dist_a, median_dist_b: 1d arrays of length bs_iter
        Boostrap distribution of medians for ``a`` and ``b``.

    See Also
    --------
    `<https://en.wikipedia.org/wiki/Bootstrapping_(statistics)>`_

    Notes
    -----
    From a programmatical point of view, it would have been better
    to implement this method for just one input array (because of
    redundant code). However, due to historical reasons (testing
    and comparability to Shape-Out 1), bootstrapping is done
    interleaved for the two arrays.
    """
    # Seed random numbers that are reproducible on different machines
    prng_object = np.random.RandomState(rs)
    # Initialize median arrays
    median_a = np.zeros(bs_iter)
    median_b = np.zeros(bs_iter)
    # If this loop is still too slow, we could get rid of it and
    # do everything with arrays. Depends on whether we will
    # eventually run into memory problems with array sizes
    # of y*bs_iter and yR*bs_iter.
    lena = len(a)
    lenb = len(b)
    for q in range(bs_iter):
        # Compute random indices and draw from a, b
        draw_a_idx = prng_object.randint(0, lena, lena)
        median_a[q] = np.median(a[draw_a_idx])
        draw_b_idx = prng_object.randint(0, lenb, lenb)
        median_b[q] = np.median(b[draw_b_idx])
    return median_a, median_b
