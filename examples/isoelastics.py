"""Plotting isoelastics

This example illustrates how to plot dclab isoelastics by reproducing
figure 3 (lower left) of :cite:`Mokbel2017`.
"""
import matplotlib.pylab as plt
from matplotlib import cm
import numpy as np
import dclab


kwargs = {"col1": "area_um",
          "col2": "deform",
          "channel_width": 20,
          "flow_rate": 0.04,
          "viscosity": 15,
          "add_px_err": False,
          "px_um": 0.2}

analy = dclab.isoelastics.default.get(method="analytical", **kwargs)
numer = dclab.isoelastics.default.get(method="numerical", **kwargs)

# plot the properties of `qpi`
plt.figure(figsize=(8, 4))

ax1 = plt.subplot(111, title="elastic sphere isoelasticity lines")


colors = [ cm.get_cmap("jet")(x) for x in np.linspace(0, 1, len(analy)) ]


for aa, nn, cc in zip(analy, numer, colors):
    ax1.plot(aa[:, 0], aa[:, 1], color=cc)
    ax1.plot(nn[:, 0], nn[:, 1], color=cc, ls=":")

ax1.set_xlim(50, 240)
ax1.set_ylim(0, 0.02)

plt.tight_layout()
plt.show()
