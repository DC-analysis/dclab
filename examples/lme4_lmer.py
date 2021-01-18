"""lme4: Linear mixed-effects models

We would like to quantify the difference between human skeletal stem cells
(SSC) and the human osteosarcoma cell line MG-63 (which is often used as a
model system for SSCs) using a likelihood ratio test based on LMM.

This example illustrates a basic LMM analysis. The data are loaded
from DCOR (:cite:`FigshareLMM`, `DCOR:figshare-11662773-v2
<https://dcor.mpl.mpg.de/dataset/figshare-11662773-v2>`_).
We treat SSC as our "treatment" and MG-63 as our "control" group.
These are just names that remind us that we are comparing one type of sample
against another type.

We are interested in the p-value, which is 0.01256 for
deformation. We repeat the analysis with area (0.0002183) and Young's
modulus (0.0002771). The p-values indicate that MG-63 (mean elastic
modulus 1.26 kPa) cells are softer than SSCs (mean elastic modulus 1.54 kPa).
The figure reproduces the last subplot of figure 6b im :cite:`Herbig2018`.
"""
import dclab
from dclab import lme4

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# https://dcor.mpl.mpg.de/dataset/figshare-11662773-v2
# SSC_16uls_rep1_20150611.rtdc
ds_ssc_rep1 = dclab.new_dataset("86cc5a47-364b-cf58-f9e3-cc114dd38e55")
# SSC_16uls_rep2_20150611.rtdc
ds_ssc_rep2 = dclab.new_dataset("ab95c914-0311-6a46-4eba-8fabca7d27d6")
# MG63_pure_16uls_rep1_20150421.rtdc
ds_mg63_rep1 = dclab.new_dataset("42cb33d4-2f7c-3c22-88e1-b9102d64d7e9")
# MG63_pure_16uls_rep2_20150422.rtdc
ds_mg63_rep2 = dclab.new_dataset("a4a98fcb-1de1-1048-0efc-b0a84d4ab32e")
# MG63_pure_16uls_rep3_20150422.rtdc
ds_mg63_rep3 = dclab.new_dataset("0a8096ce-ea7a-e36d-1df3-42c7885cd71c")

datasets = [ds_ssc_rep1, ds_ssc_rep2, ds_mg63_rep1, ds_mg63_rep2, ds_mg63_rep3]
for ds in datasets:
    # perform filtering
    ds.config["filtering"]["area_ratio min"] = 0
    ds.config["filtering"]["area_ratio max"] = 1.05
    ds.config["filtering"]["area_um min"] = 120
    ds.config["filtering"]["area_um max"] = 550
    ds.config["filtering"]["deform min"] = 0
    ds.config["filtering"]["deform max"] = 0.1
    ds.apply_filter()
    # enable computation of Young's modulus
    ds.config["calculation"]["emodulus lut"] = "LE-2D-FEM-19"
    ds.config["calculation"]["emodulus medium"] = "CellCarrier"
    ds.config["calculation"]["emodulus temperature"] = 23.0

# setup lme4 analysis
rlme4 = lme4.Rlme4(model="lmer")
rlme4.add_dataset(ds_ssc_rep1, group="treatment", repetition=1)
rlme4.add_dataset(ds_ssc_rep2, group="treatment", repetition=2)
rlme4.add_dataset(ds_mg63_rep1, group="control", repetition=1)
rlme4.add_dataset(ds_mg63_rep2, group="control", repetition=2)
rlme4.add_dataset(ds_mg63_rep3, group="control", repetition=3)

# perform analysis for deformation
for feat in ["area_um", "deform", "emodulus"]:
    res = rlme4.fit(feature=feat)
    print("Results for {}:".format(feat))
    print("  p-value", res["anova p-value"])
    print("  mean of MG-63", res["fixed effects intercept"])
    print("  fixed effect size", res["fixed effects treatment"])

# prepare for plotting
df = pd.DataFrame()
for ds in datasets:
    group = ds.config["experiment"]["sample"].split()[0]
    rep = ds.config["experiment"]["sample"].split()[-1]
    dfi = pd.DataFrame.from_dict(
        {"area_m": ds["area_um"][ds.filter.all],
         "deform": ds["deform"][ds.filter.all],
         "emodulus": ds["emodulus"][ds.filter.all],
         "group and repetition": [group + " " + rep] * ds.filter.all.sum(),
         "group": [group] * ds.filter.all.sum(),
         })
    df = df.append(dfi)

# plot
fig = plt.figure(figsize=(8, 5))
ax = sns.boxplot(x="group and repetition", y="emodulus", data=df, hue="group")
# note that `res` is still the result for "emodulus"
numstars = sum([res["anova p-value"] < .05,
                res["anova p-value"] < .01,
                res["anova p-value"] < .001,
                res["anova p-value"] < .0001])
# significance bars
h = .1
y1 = 6
y2 = 4.2
y3 = 6.2
ax.plot([-.5, -.5, 1, 1], [y1, y1+h, y1+h, y1], lw=1, c="k")
ax.plot([2, 2, 4.5, 4.5], [y2, y2+h, y2+h, y2], lw=1, c="k")
ax.plot([.25, .25, 3.25, 3.25], [y1+h, y1+2*h, y1+2*h, y2+h], lw=1, c="k")
ax.text(2, y3, "*"*numstars, ha='center', va='bottom', color="k")
ax.set_ylim(0, 7)

plt.tight_layout()
plt.show()
