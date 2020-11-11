"""lme4: Generalized linear mixed-effects models with differential deformation

This example illustrates how to perform a differential feature
(including reservoir data) GLMM analysis. The example data
are taken from DCOR (:cite:`FigshareLMM`, `DCOR:figshare-11662773-v2
<https://dcor.mpl.mpg.de/dataset/figshare-11662773-v2>`_).
As in the :ref:`previous example <example_lme4_lmer>`, we treat SSC
as our "treatment" and MG-63 as our "control" group.

The p-value for the differential deformation is magnitudes lower than
the p-value for the (non-differential) deformation in the previous example.
This indicates that there is a non-negligible initial deformation of the
cells in the reservoir.
"""
from dclab import lme4, new_dataset

# https://dcor.mpl.mpg.de/dataset/figshare-11662773-v2
datasets = [
    # SSC channel
    [new_dataset("86cc5a47-364b-cf58-f9e3-cc114dd38e55"), "treatment", 1],
    [new_dataset("ab95c914-0311-6a46-4eba-8fabca7d27d6"), "treatment", 2],
    # SSC reservoir
    [new_dataset("761ab515-0416-ede8-5137-135c1682580c"), "treatment", 1],
    [new_dataset("3b83d47b-d860-4558-51d6-dcc524f5f90d"), "treatment", 2],
    # MG-63 channel
    [new_dataset("42cb33d4-2f7c-3c22-88e1-b9102d64d7e9"), "control", 1],
    [new_dataset("a4a98fcb-1de1-1048-0efc-b0a84d4ab32e"), "control", 2],
    [new_dataset("0a8096ce-ea7a-e36d-1df3-42c7885cd71c"), "control", 3],
    # MG-63 reservoir
    [new_dataset("56c449bb-b6c9-6df7-6f70-6744b9960980"), "control", 1],
    [new_dataset("387b5ac9-1cc6-6cac-83d1-98df7d687d2f"), "control", 2],
    [new_dataset("7ae49cd7-10d7-ef35-a704-72443bb32da7"), "control", 3],
]

# perform filtering
for ds, _, _ in datasets:
    ds.config["filtering"]["area_ratio min"] = 0
    ds.config["filtering"]["area_ratio max"] = 1.05
    ds.config["filtering"]["area_um min"] = 120
    ds.config["filtering"]["area_um max"] = 550
    ds.config["filtering"]["deform min"] = 0
    ds.config["filtering"]["deform max"] = 0.1
    ds.apply_filter()

# perform LMM analysis for differential deformation
# setup lme4 analysis
rlme4 = lme4.Rlme4(feature="deform")
for ds, group, repetition in datasets:
    rlme4.add_dataset(ds, group=group, repetition=repetition)

# LMM
lmer_result = rlme4.fit(model="lmer")
print("LMM p-value", lmer_result["anova p-value"])  # 0.00000351

# GLMM with log link function
glmer_result = rlme4.fit(model="glmer+loglink")
print("GLMM p-value", glmer_result["anova p-value"])  # 0.000868
