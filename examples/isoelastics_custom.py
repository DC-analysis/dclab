"""Plotting custom isoelastics

This example illustrates how to extract custom isoelasticity lines from
the dclab look-up tables by reproducing figure 3 (right) of
:cite:`Wittwer2022`.

Note that at the boundary of the support of a look-up table, the isoelasticity
lines may break away in perpendicular directions. The underlying reason
is that the look-up table is first mapped onto a grid from which the
constant isoelasticity lines are extracted. Since the Young's modulus values
are linearly interpolated from the LUT onto that grid, there can be
inaccuracies for pixels that are at the LUT boundary.

An elaborate way of getting rid of these inaccuracies (and this is
how the isoelasticity lines for dclab are extracted), is to extend
the LUT by fitting a polynomial to isoelasticity lines which are well-defined
within the LUT and extrapolating these lines beyond the boundary of the LUT.
This technique is documented in the `scripts` directory of the dclab
repository.

A quicker and much less elaborate way of getting around this
issue is to simply crop the individual isoelasticity lines where necessary.
"""
import matplotlib.pylab as plt
import numpy as np
import skimage

import dclab
from dclab.features import emodulus


colors = ["r", "b"]
linestyles = [":", "-"]

plt.figure(figsize=(8, 4))
ax = plt.subplot(111,
                 title="Comparison of the isoelasticity lines of two LUTs")

grid_sie = 250

for ii, lut_name in enumerate(["LE-2D-FEM-19", "HE-3D-FEM-22"]):
    area_um = np.linspace(0, 350, grid_sie, endpoint=True)
    deform = np.linspace(0, 0.2, grid_sie, endpoint=True)
    area_um_grid, deform_grid = np.meshgrid(area_um, deform, indexing="ij")

    emod = emodulus.get_emodulus(area_um=area_um_grid,
                                 deform=deform_grid,
                                 medium=6.0,
                                 channel_width=20,
                                 flow_rate=0.04,
                                 px_um=0,
                                 temperature=None,
                                 visc_model=None,
                                 lut_data=lut_name)

    levels = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 6.0]
    for level in levels:
        conts = skimage.measure.find_contours(emod, level=level)
        if not conts:
            continue
        # get the longest one
        idx = np.argmax([len(cc) for cc in conts])
        cc = conts[idx]
        # remove nan values
        cc = cc[~np.isnan(np.sum(cc, axis=1))]
        # scale isoelastics back
        cc_sc = np.copy(cc)
        cc_sc[:, 0] = cc[:, 0] / grid_sie * 350
        cc_sc[:, 1] = cc[:, 1] / grid_sie * 0.2
        plt.plot(cc_sc[:, 0], cc_sc[:, 1],
                 color=colors[ii],
                 ls=linestyles[ii],
                 label=lut_name if level == levels[0] else None)

ax.set_ylim(-0.005, 0.1)
ax.set_xlabel(dclab.dfn.get_feature_label("area_um"))
ax.set_ylabel(dclab.dfn.get_feature_label("deform"))
plt.legend()
plt.tight_layout()
plt.show()
