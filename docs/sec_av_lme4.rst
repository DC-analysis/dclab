.. _sec_av_lme4:

===========================
Linear mixed-effects models
===========================
It is not straightforward to define a p-Value for RT-DC data
(e.g. change in deformation for a treatment vs. its control).
This is somewhat counter-intuitive, because one could assume
that the large number of events in a single dataset should be
enough to compare two datasets. However, Focus changes, chip-to-chip
variations, etc. may generate systematic offsets which make a
direct comparison (e.g. t-Test) impossible. Linear mixed effect
models (LMM) allow to assign a significance to a treatment measurement
compared to a control measuerement (fixed effect) while considering the
systematic bias in-between the measurement repetitions (random effect).

dclab offers LMM analysis as described in :cite:`Herbig2018`.
The LMM analysis is performed using the `lme4
<https://github.com/lme4/lme4>`_ R package.


Computing p-values with lme4 in dclab
=====================================

dclab exposes two models from lme4:

- **linear mixed-effects models** ("lmer"): This is basically the simplest
  way of determining whether or not a treatment has an effect.
- **generalized linear mixed-effects models with a log-link function** ("glmer+loglink"):
  This model makes use of lme4's generalized linear effects model (GLMM)
  ``glmer`` function with a log-link function (``family=Gamma(link='log')``).
  This is used for data that is log-normally distributed. Log-normal behaviour
  is quite common, especially in biology. When a physical parameter has a
  lower limit, and the measured values are close to that limit, the
  resulting distribution will be skewed, resembling a log-normal distribution.
  In case of RT-DC this is specially (but not only) true for deformation.
  Another example is area, which also has a lower limit of zero and may
  therefore have a skewed distribution. While GLMMs are designed to handle
  skewed data, it was shown that LMMs already deliver robust results, even
  for highly skewed data :cite:`gelman_hill_2006`.

.. warning::
    The decision whether to use LMM or GLMM is not
    particularly important. Ideally, both LMM and GLMM are consistent.
    However, never perform both analyses only to then pick the one
    with the lowest p-value. This is p-hacking! The analysis routine
    should be defined beforehand. If in doubt, stick to LMM.

An LMM analysis is straight-forward in dclab:

.. code:: python

    import dclab
    from dclab import lme4

    # Load the data
    ds_rep1_ctl = dclab.new_dataset(...)  # control measurement, 1st repetition
    ds_rep1_trt = dclab.new_dataset(...)  # treatment measurement, 1st repetition
    ds_rep2_ctl = dclab.new_dataset(...)  # control measurement, 2nd repetition
    ds_rep2_trt = dclab.new_dataset(...)  # treatment measurement, 2nd repetition

    # Instantiate Rlme4
    rlme4 = lme4.Rlme4(model="lmer", feature="deform")

    # Add the datasets
    rlme4.add_dataset(ds=ds_rep1_ctl, group="control", repetition=1)
    rlme4.add_dataset(ds=ds_rep1_trt, group="treatment", repetition=1)
    rlme4.add_dataset(ds=ds_rep2_ctl, group="control", repetition=2)
    rlme4.add_dataset(ds=ds_rep2_ctl, group="treatment", repetition=2)

    # Perform the analysis
    result = rlme4.fit()
    print("p-value:", result["anova p-value"])
    print("fixed effect:", result["fixed effects treatment"])
    print("model converged:", result["model converged"])

The ``fit()`` function returns the most important results and also exposes
some of the underlying R objects (see :func:`dclab.lme4.wrapr.Rlme4.fit`).
An LMM example is also given in the :ref:`example section <example_lme4_lmer>`.

.. note::
    If a treatment and a control share the same repetition number, it
    is implied that they are paired. For those measurements, lme4 will
    perform a paired test. In your experimental design you determine
    which measurements are paired, before doing any experiments. Pairing
    can be done e.g. for measurements done on the same day or on the
    same chip. In cases where you perform the control measurements on
    one day and the treatment measurements on another day, you could
    still pair them. Just keep in mind that this could introduce
    systematic errors, if the measurement conditions (temperature,
    illumination, etc.) were not identical. Under no circumstances,
    choose a pairing that yields the lowest p-value (p-hacking).

    Alternatively, you can also run an unpaired test by just giving
    each measurement a different repetition number. For example for
    3x control and 3x treatment measurements, you could enumerate the
    repetition number from 1 to 6.


Differential feature analysis with reservoir data
=================================================
The (G)LMM analysis is only applicable if the feature chosen is not pronounced
visibly in the reservoir measurements. For instance, if a treatment results
in a significant change in deformation already in the reservoir, then the
p-value determined for the channel data might be underestimated (too many
stars). In this case, the information of the reservoir measurement
must be included by means of differential deformation :cite:`Herbig2018`.
The idea of differential deformation is to subtract the reservoir from the
channel deformation. Since it is not possible to assign the events in the
reservoir to the events in the channel (two different measurements),
bootstrapping is employed which generates statistical representations
of the two measurements that can then be subtracted from one
another. Then, for the actual LMM analysis, only the differential
deformation is used.

To perform a differential feature analysis, simply add the reservoir
measurements to the :class:`dclab.lme4.wrapr.Rlme4` class (they are
recognized as reservoir measurements via their meta data).

.. code:: python

    # Load the data
    ds_rep1_ctl = dclab.new_dataset(...)  # control measurement, 1st repetition (channel)
    ds_rep1_ctl_res = dclab.new_dataset(...)  # control measurement, 1st repetition (reservoir)
    [...]

    # Instantiate Rlme4
    rlme4 = lme4.Rlme4(model="lmer", feature="deform")

    # Add the datasets
    rlme4.add_dataset(ds=ds_rep1_ctl, group="control", repetition=1)
    rlme4.add_dataset(ds=ds_rep1_ctl_res, group="control", repetition=1)
    [...]

    # Perform the analysis
    result = rlme4.fit()
    assert results["is differential"]  # adding "reservoir" data forces differential analysis

Keep in mind that the analysis is now performed using the differential
features and not the actual features (``result["is differential"]``).
For more information, please see :func:`dclab.lme4.wrapr.Rlme4.get_differential_dataset`
and :func:`dclab.lme4.wrapr.bootstrapped_median_distributions`.
A full example, including GLMM and differential deformation, is given in the
:ref:`example section <example_lme4_glmer_diff>`.
