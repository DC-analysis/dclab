"""R lme4 wrapper"""
import logging
import numbers
import pathlib
import tempfile

import importlib_resources
import numpy as np

from .. import definitions as dfn
from ..rtdc_dataset.core import RTDCBase

from . import rsetup


logger = logging.getLogger(__name__)


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

        self.set_options(model=model, feature=feature)

        # Make sure that lme4 is available
        if not rsetup.has_lme4():
            logger.info("Installing lme4, this may take a while!")
            rsetup.require_lme4()

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
        - For each repetition, there must be a "treatment" (``1``) and a
          "control" (``0``) group.
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

        - model: "feature ~ group + (1 + group | repetition)"
          (random intercept + random slope model)
        - the null model: "feature ~ (1 + group | repetition)"
          (without the fixed effect introduced by the "treatment" group).

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

            - "anova p-value": Anova likelihood ratio test (significance)
            - "feature": name of the feature used for the analysis
              ``self.feature``
            - "fixed effects intercept": Mean of ``self.feature`` for all
              controls; In the case of the "glmer+loglink" model, the intercept
              is already back transformed from log space.
            - "fixed effects treatment": The fixed effect size between the mean
              of the controls and the mean of the treatments relative to
              "fixed effects intercept"; In the case of the "glmer+loglink"
              model, the fixed effect is already back transformed from log
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
            - "r model summary": Summary of the model
            - "r model coefficients": Model coefficient table
            - "r script": the R script used
            - "r output": full output of the R script
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

        # concatenate and populate arrays for R
        features_c = np.concatenate(features)
        groups_c = np.zeros(len(features_c), dtype=str)
        repetitions_c = np.zeros(len(features_c), dtype=int)
        pos = 0
        for ii in range(len(features)):
            size = len(features[ii])
            groups_c[pos:pos+size] = groups[ii][0]
            repetitions_c[pos:pos+size] = repetitions[ii]
            pos += size

        # Run R with the given template script
        rscript = importlib_resources.read_text("dclab.lme4",
                                                "lme4_template.R")
        _, script_path = tempfile.mkstemp(prefix="dclab_lme4_", suffix=".R",
                                          text=True)
        script_path = pathlib.Path(script_path)
        rscript = rscript.replace("<MODEL_NAME>", self.model)
        rscript = rscript.replace("<FEATURES>", arr2str(features_c))
        rscript = rscript.replace("<REPETITIONS>", arr2str(repetitions_c))
        rscript = rscript.replace("<GROUPS>", arr2str(groups_c))
        script_path.write_text(rscript, encoding="utf-8")

        result = rsetup.run_command((rsetup.get_r_script_path(), script_path))

        ret_dict = self.parse_result(result)
        ret_dict["is differential"] = self.is_differential()
        ret_dict["feature"] = self.feature
        ret_dict["r script"] = rscript
        ret_dict["r output"] = result
        assert ret_dict["model"] == self.model

        return ret_dict

    def get_differential_dataset(self):
        """Return the differential dataset for channel/reservoir data

        The most famous use case is differential deformation. The idea
        is that you cannot tell what the difference in deformation
        from channel to reservoir, because you never measure the
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

    def parse_result(self, result):
        resd = result.split("OUTPUT")
        ret_dict = {}
        for item in resd:
            string = item.split("#*#")[0]
            key, value = string.split(":", 1)
            key = key.strip()
            value = value.strip().replace("\n\n", "\n")

            if key == "fixed effects repetitions":
                rows = value.split("\n")[1:]
                reps = []
                for row in rows:
                    reps.append([float(vv) for vv in row.split()[1:]])
                value = np.array(reps).transpose()
            elif key == "model converged":
                value = value == "TRUE"
            elif value == "NA":
                value = np.nan
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass

            ret_dict[key] = value
        return ret_dict

    def set_options(self, model=None, feature=None):
        """Set analysis options"""
        if model is not None:
            assert model in ["lmer", "glmer+loglink"]
            self.model = model
        if feature is not None:
            assert dfn.scalar_feature_exists(feature)
            self.feature = feature


def arr2str(a):
    """Convert an array to a string"""
    if isinstance(a.dtype.type, np.integer):
        return ",".join(str(dd) for dd in a.tolist())
    elif a.dtype.type == np.str_:
        return ",".join(f"'{dd}'" for dd in a.tolist())
    else:
        return ",".join(f"{dd:.16g}" for dd in a.tolist())


def bootstrapped_median_distributions(a, b, bs_iter=1000, rs=117):
    """Compute the bootstrapped distributions for two arrays.

    Parameters
    ----------
    a, b: 1d ndarray of length N
        Input data
    bs_iter: int
        Number of bootstrapping iterations to perform
        (output size).
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
    From a programmatic point of view, it would have been better
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
