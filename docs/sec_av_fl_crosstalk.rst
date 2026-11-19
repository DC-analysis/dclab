.. _sec_av_fl_crosstalk:

===============================
Fluorescence crosstalk
===============================
The emission spectra of fluorescent dyes are broad. Therefore, a dye
that is detected in one fluorescence channel (e.g. FL-1) usually also
contributes to the signal of other channels (e.g. FL-2). This
crosstalk (also called spillover or bleed-through) can be corrected for,
if the fraction of signal that spills from one channel into another is
known. In dclab, the crosstalk-corrected maximum fluorescence signals
are available as the ancillary features ``fl1_max_ctc``, ``fl2_max_ctc``,
and ``fl3_max_ctc``.


Theory
======
The crosstalk coefficient :math:`c_{ij}` is the fraction of the signal of
a dye detected in channel :math:`i` that is additionally detected in
channel :math:`j`. The measured signal :math:`F_j` of channel :math:`j`
is the sum of the actual (crosstalk-free) signal :math:`F'_j` of the
dye detected in that channel and the crosstalk from all other channels:

.. math::

    F_j = F'_j + \sum_{i \neq j} c_{ij} F'_i

For three channels, this can be written in matrix form with the
spillover matrix :math:`\mathbf{C}`, whose diagonal elements are one:

.. math::

    \begin{pmatrix} F_1 & F_2 & F_3 \end{pmatrix} =
    \begin{pmatrix} F'_1 & F'_2 & F'_3 \end{pmatrix}
    \begin{pmatrix}
      1 & c_{12} & c_{13} \\
      c_{21} & 1 & c_{23} \\
      c_{31} & c_{32} & 1
    \end{pmatrix}

The crosstalk-corrected signals :math:`F'_j` are obtained by
multiplying the measured signals with the inverse of :math:`\mathbf{C}`
(the compensation matrix, see
:func:`dclab.features.fl_crosstalk.get_compensation_matrix`).
For two channels, all coefficients to and from the third channel
are zero.


Determining the crosstalk coefficients
======================================
The crosstalk coefficients depend on the dyes, the optical filters,
and the detector settings of the setup. They have to be determined
with control samples that are stained with only one dye
(single-stained controls), using the same setup and settings as
for the actual measurement.
For a sample stained with a dye detected in channel :math:`i`, the
crosstalk coefficient to channel :math:`j` is the ratio

.. math::

    c_{ij} = \frac{F_j}{F_i}

of the signals measured in channel :math:`j` and channel :math:`i`,
e.g. computed from the median values of the stained population.
If unstained cells already yield a signal (offset) in these channels,
this offset should be subtracted from :math:`F_i` and :math:`F_j`
before computing the ratio.

To check the result, apply the crosstalk correction to the
single-stained controls: The population stained with the dye of channel
:math:`i` should, after correction, have the same signal in
channel :math:`j` as unstained cells.


Crosstalk correction in dclab
=============================
The crosstalk coefficients are set in the "calculation" section of the
dataset configuration (see :ref:`sec_analysis_meta`), where e.g.
``"crosstalk fl12"`` is the coefficient :math:`c_{12}` (crosstalk from
channel 1 to channel 2):

.. ipython::

    In [1]: import dclab

    In [2]: ds = dclab.new_dataset("data/example_traces.rtdc")

    In [3]: ds.config["calculation"]["crosstalk fl12"] = 0.05

    In [4]: ds.config["calculation"]["crosstalk fl13"] = 0.01

    In [5]: ds.config["calculation"]["crosstalk fl21"] = 0.10

    In [6]: ds.config["calculation"]["crosstalk fl23"] = 0.02

    In [7]: ds.config["calculation"]["crosstalk fl31"] = 0.00

    In [8]: ds.config["calculation"]["crosstalk fl32"] = 0.03

    In [9]: ds["fl1_max"][:4]

    In [10]: ds["fl1_max_ctc"][:4]

The coefficients in this example are arbitrary.

- If a dataset has all three fluorescence channels (``fl1_max``,
  ``fl2_max``, and ``fl3_max``), all six coefficients must be set.
  Otherwise, a
  ``MissingCrosstalkMatrixElementsError``
  is raised when accessing the corrected features.
- If a dataset has only two fluorescence channels, only the two
  coefficients between these channels are required and used
  (e.g. ``"crosstalk fl12"`` and ``"crosstalk fl21"``). Coefficients
  involving the missing channel are ignored.
- The corrected features are computed when they are first accessed
  and cached. When you change a coefficient, they are recomputed.

For data that are not stored in a dataset, you can use
:func:`dclab.features.fl_crosstalk.correct_crosstalk` directly.


Negative values after correction
================================
The correction subtracts the crosstalk from the measured signal.
Because of measurement noise, the corrected signal of events
whose actual signal in a channel is (close to) zero is distributed
around zero, i.e. some corrected values are negative.
This is expected: For a correct crosstalk correction, the corrected
signal of a population that is negative for a dye should be
centered around the signal of unstained cells (zero, if there is no
offset).

Please note that events with negative (or zero) values cannot be
shown on a logarithmic axis. In scatter plots with logarithmic axes,
these events are not displayed, i.e. fewer events are shown than
there are in the dataset (the event count of filtered data is not
affected). Use linear axes to see all events, or check whether
populations are cut off at zero before drawing conclusions from
logarithmic plots.

The following example with simulated data (no real measurement)
illustrates crosstalk from channel 1 into channel 2
(:math:`c_{12} = 0.2`) for three populations: unstained, stained with
the dye of channel 1, and stained with the dye of channel 2.

.. plot::

    import matplotlib.pylab as plt
    import numpy as np

    import dclab

    rng = np.random.default_rng(42)
    size = 500
    # actual (crosstalk-free) signals of three populations:
    # unstained, stained with dye 1, stained with dye 2
    fl1_act = np.concatenate([np.zeros(size),
                              rng.normal(1000, 150, size),
                              np.zeros(size)])
    fl2_act = np.concatenate([np.zeros(size),
                              np.zeros(size),
                              rng.normal(600, 100, size)])
    # measured signals with crosstalk from channel 1 to channel 2
    # and detector noise
    c12 = 0.2
    fl1 = fl1_act + rng.normal(0, 15, 3 * size)
    fl2 = fl2_act + c12 * fl1_act + rng.normal(0, 15, 3 * size)

    ds = dclab.new_dataset({"fl1_max": fl1, "fl2_max": fl2})
    ds.config["calculation"]["crosstalk fl12"] = c12
    ds.config["calculation"]["crosstalk fl21"] = 0

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    axes[0].plot(ds["fl1_max"], ds["fl2_max"], ".", ms=2)
    axes[0].set_title("measured")
    axes[0].set_xlabel("fl1_max [a.u.]")
    axes[0].set_ylabel("fl2_max [a.u.]")
    axes[1].plot(ds["fl1_max_ctc"], ds["fl2_max_ctc"], ".", ms=2)
    axes[1].set_title("crosstalk-corrected")
    axes[1].set_xlabel("fl1_max_ctc [a.u.]")
    axes[1].set_ylabel("fl2_max_ctc [a.u.]")
    for ax in axes:
        ax.grid()
    plt.tight_layout()
    plt.show()

Before the correction, the population stained with dye 1 has a
signal in channel 2 of about 20 % of its signal in channel 1. After the
correction, it is located at the same signal in channel 2 as the
unstained population, around zero, including negative values.
