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


class Rlme4():
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

        References
        ----------
        .. [1] R package "lme4":
               Bates D, Maechler M, Bolker B and Walker S (2015). lme4:
               Linear mixed- effects models using Eigen and S4. R package
               version 1.1-9, https://CRAN.R-project.org/package=lme4.

        .. [2] R function "anova" from package "stats":
               Chambers, J. M. and Hastie, T. J. (1992) Statistical Models
               in S, Wadsworth & Brooks/Cole
        """
        #: modeling method to use (lmer or glmer)
        self.model = None
        #: dclab feature for which to perform the analysis
        self.feature = None
        #: list of [RTDCBase, column, repetition, chip_region]
        self.data = []

        #: modeling function
        self.r_func_model = "feature ~ group + (1 + group | repetition)"
        self.r_func_nullmodel = "feature ~ (1 + group | repetition)"

        self.set_options(model=model, feature=feature)

        # Make sure that lme4 is available
        if not rsetup.has_lme4():
            warnings.warn("Installing lme4, this may take a while!",
                          Lme4InstallWarning)
            rsetup.install_lme4()

    def add_dataset(self, ds, group, repetition):
        assert group in ["treatment", "control"]
        assert isinstance(ds, RTDCBase)
        assert isinstance(repetition, numbers.Integral)
        self.data.append([ds, group, repetition,
                          ds.config["setup"]["chip region"]])

    def check_data(self):
        # Check that we have enough data
        if len(self.data) < 3:
            msg = "Linear Mixed Models require repeated measurements. " + \
                  "Please select more treatment repetitions."
            raise ValueError(msg)

    def set_options(self, model=None, feature=None):
        if model is not None:
            assert model in ["lmer", "glmer+loglink"]
            self.model = model
        if feature is not None:
            assert dfn.scalar_feature_exists(feature)
            self.feature = feature

    def bootstrap(self, features, groups, repetitions, regions):
        raise NotImplementedError("Boostrapping not yet implemented!")

    def fit(self, model=None, feature=None):
        """Perform G(LMM) fit

        Parameters
        ----------
        model: str (optional)
            One of:

            - "lmer": linear mixed model using lme4's ``lmer``
            - "glmer+loglink": generalized linear mixed model using
              lme4's ``glmer`` with an additional a log-link function
              via the ``family=Gamma(link='log'))`` keyword.
        feature: str (optional)
            Dclab feature for which to compute the model

        Returns
        -------
        results: dict
            The results of the entire fitting process:

            - "fxed effects intercept": Mean of ``feature`` for all controls
            - "fixed effects treatment": The fixed effect size between the mean
              of the controls and the mean of the treatments
              relative to "fixed effects intercept"
            - "anova p-value": Anova likelyhood ratio test (significance)
            - "model summary": Summary of the model
            - "model coefficients": Model coefficient table
            - "r_err": errors and warnings from R
            - "r_out": standard output from R
        """

        self.set_options(model=model, feature=feature)
        self.check_data()

        # Assemble data
        features = []
        groups = []
        regions = []
        repetitions = []
        for dd in self.data:
            fdata = dd[0][self.feature]
            # remove nan/inf
            fdata = fdata[~np.logical_or(np.isnan(fdata), np.isinf(fdata))]
            features.append(fdata)
            groups.append(dd[1])
            repetitions.append(dd[2])
            regions.append(dd[3])

        # perform bootstrapping with reservoir data if applicable
        if "reservoir" in regions:
            features, groups, repetitions = self.bootstrap(
                features, groups, repetitions, regions)

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

            # Anova analysis
            anova = r["anova"](r_model, r_nullmodel, test="Chisq")
            pvalue = anova.rx["Pr(>Chisq)"][0][1]
            model_summary = r["summary"](r_model)
            coeff_summary = r["coef"](r_model)

            coeffs = r["data.frame"](r["coef"](model_summary))

            # TODO: find out what p.normal are for:
            # rpy2.robjects.globalenv["model"] = r_model
            # r("coefs <- data.frame(coef(summary(model)))")
            # r("coefs$p.normal=2*(1-pnorm(abs(coefs$t.value)))")

            fe_icept = coeffs[0][0]
            fe_treat = coeffs[1][0]
            if self.model == "glmer+loglink":
                # transform back from log
                fe_icept = np.exp(fe_icept)
                fe_treat = np.exp(fe_icept + fe_treat) - np.exp(fe_icept)

        ret_dict = {
            "fixed effects intercept": fe_icept,
            "fixed effects treatment": fe_treat,  # aka "fixed effect"
            "anova p-value": pvalue,
            "summary_model": model_summary,
            "summary_coefficients": coeff_summary,
            "r_err": ac.get_warnerrors(),
            "r_out": ac.get_prints(),
        }
        return ret_dict
