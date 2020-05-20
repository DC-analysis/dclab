"""Dataset overview plot

This example demonstrates basic data visualization with dclab and
matplotlib. To run this script, download the reference dataset
`calibration_beads.rtdc` :cite:`FigshareRef` and place it in the
same directory.

You will find more examples in the
:ref:`advanced usage <sec_advanced_scripting>` section of this
documentation.
"""
import matplotlib.pylab as plt
import numpy as np

import dclab

# Dataset to display
DATASET_PATH = "calibration_beads.rtdc"
# Features for scatter plot
SCATTER_X = "area_um"
SCATTER_Y = "deform"
# Event index to display
EVENT_INDEX = 100

xlabel = dclab.dfn.get_feature_label(SCATTER_X)
ylabel = dclab.dfn.get_feature_label(SCATTER_Y)

ds = dclab.new_dataset(DATASET_PATH)

fig = plt.figure(figsize=(8, 7))


ax1 = plt.subplot(221, title="Simple scatter plot")
ax1.plot(ds[SCATTER_X], ds[SCATTER_Y], "o", color="k", alpha=.2, ms=1)
ax1.set_xlabel(xlabel)
ax1.set_ylabel(ylabel)
ax1.set_xlim(19, 40)
ax1.set_ylim(0.005, 0.03)

ax2 = plt.subplot(222, title="KDE scatter plot")
sc = ax2.scatter(ds[SCATTER_X], ds[SCATTER_Y],
                 c=ds.get_kde_scatter(xax=SCATTER_X,
                                      yax=SCATTER_Y,
                                      kde_type="multivariate"),
                 s=3)
plt.colorbar(sc, label="kernel density [a.u]", ax=ax2)
ax2.set_xlabel(xlabel)
ax2.set_ylabel(ylabel)
ax2.set_xlim(19, 40)
ax2.set_ylim(0.005, 0.03)

ax3 = plt.subplot(425, title="Event image with contour")
ax3.imshow(ds["image"][EVENT_INDEX], cmap="gray")
ax3.plot(ds["contour"][EVENT_INDEX][:, 0],
         ds["contour"][EVENT_INDEX][:, 1],
         c="r")
ax3.set_xlabel("Detector X [px]")
ax3.set_ylabel("Detector Y [px]")

ax4 = plt.subplot(427, title="Event mask with µm-scale")
pxsize = ds.config["imaging"]["pixel size"]
ax4.imshow(ds["mask"][EVENT_INDEX],
           extent=[0, ds["mask"].shape[2] * pxsize,
                   0, ds["mask"].shape[1] * pxsize],
           cmap="gray")
ax4.set_xlabel("Detector X [µm]")
ax4.set_ylabel("Detector Y [µm]")

ax5 = plt.subplot(224, title="Fluorescence traces")
flsamples = ds.config["fluorescence"]["samples per event"]
flrate = ds.config["fluorescence"]["sample rate"]
fltime = np.arange(flsamples) / flrate * 1e6
# here we plot "fl?_raw"; you may also plot "fl?_med"
ax5.plot(fltime, ds["trace"]["fl1_raw"][EVENT_INDEX],
         c="#15BF00", label="fl1_raw")
ax5.plot(fltime, ds["trace"]["fl2_raw"][EVENT_INDEX],
         c="#BF8A00", label="fl2_raw")
ax5.plot(fltime, ds["trace"]["fl3_raw"][EVENT_INDEX],
         c="#BF0C00", label="fl3_raw")
ax5.legend()
ax5.set_xlim(ds["fl1_pos"][EVENT_INDEX] - 2*ds["fl1_width"][EVENT_INDEX],
             ds["fl1_pos"][EVENT_INDEX] + 2*ds["fl1_width"][EVENT_INDEX])
ax5.set_xlabel("Event time [µs]")
ax5.set_ylabel("Fluorescence [a.u.]")

plt.tight_layout()

plt.show()
