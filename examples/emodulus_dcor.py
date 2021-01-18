"""Young's modulus computation from data on DCOR

This example reproduces the lower right subplot of figure 10
in :cite:`Herold2017`. It illustrates how the Young's modulus
of elastic beads can be retrieved correctly (independent of
the flow rate, with correction for pixelation and shear-thinning)
using the area-deformation look-up table implemented in dclab
(right plot). For comparison, the flow-rate-dependent deformation
is also shown (left plot).

`The dataset <https://dcor.mpl.mpg.de/dataset/figshare-12721436-v1>`_
is loaded directly from `DCOR <https://dcor.mpl.mpg.de>`_ and thus
an active internet connection is required for this example.
"""
import dclab
import matplotlib.pylab as plt

# The dataset is also available on figshare
# (https://doi.org/10.6084/m9.figshare.12721436.v1), but we
# are accessing it through the DCOR API, because we do not
# have the time to download the entire dataset. The dataset
# name is figshare-12721436-v1. These are the resource IDs:
ds_loc = ["e4d59480-fa5b-c34e-0001-46a944afc8ea",
          "2cea205f-2d9d-26d0-b44c-0a11d5379152",
          "2cd67437-a145-82b3-d420-45390f977a90",
          ]
ds_list = []  # list of opened datasets
labels = []  # list of flow rate labels

# load the data
for loc in ds_loc:
    ds = dclab.new_dataset(loc)
    labels.append("{:.2f}".format(ds.config["setup"]["flow rate"]))
    # emodulus computation
    ds.config["calculation"]["emodulus lut"] = "LE-2D-FEM-19"
    ds.config["calculation"]["emodulus medium"] = ds.config["setup"]["medium"]
    ds.config["calculation"]["emodulus temperature"] = \
        ds.config["setup"]["temperature"]
    # filtering
    ds.config["filtering"]["area_ratio min"] = 1.0
    ds.config["filtering"]["area_ratio max"] = 1.1
    ds.config["filtering"]["deform min"] = 0
    ds.config["filtering"]["deform max"] = 0.035
    # This option will remove "nan" events that appear in the "emodulus"
    # feature. If you are not working with DCOR, this might lead to a
    # longer computation time, because all available features are
    # computed locally. For data on DCOR, this computation already has
    # been done.
    ds.config["filtering"]["remove invalid events"] = True
    ds.apply_filter()
    # Create a hierarchy child for convenience reasons
    # (Otherwise we would have to do e.g. ds["deform"][ds.filter.all]
    # everytime we need to access a feature)
    ds_list.append(dclab.new_dataset(ds))

# plot
fig = plt.figure(figsize=(8, 4))

# box plot for deformation
ax1 = plt.subplot(121)
ax1.set_ylabel(dclab.dfn.get_feature_label("deform"))
data_deform = [di["deform"] for di in ds_list]
# Uncomment this line if you are not filtering invalid events (above)
# data_deform = [d[~np.isnan(d)] for d in data_deform]
bplot1 = ax1.boxplot(data_deform,
                     vert=True,
                     patch_artist=True,
                     labels=labels,
                     )

# box plot for Young's modulus
ax2 = plt.subplot(122)
ax2.set_ylabel(dclab.dfn.get_feature_label("emodulus"))
data_emodulus = [di["emodulus"] for di in ds_list]
# Uncomment this line if you are not filtering invalid events (above)
# data_emodulus = [d[~np.isnan(d)] for d in data_emodulus]
bplot2 = ax2.boxplot(data_emodulus,
                     vert=True,
                     patch_artist=True,
                     labels=labels,
                     )

# colors
colors = ["#0008A5", "#A5008D", "#A50100"]
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

# axes
for ax in [ax1, ax2]:
    ax.grid()
    ax.set_xlabel("flow rate [µL/s]")

plt.tight_layout()
plt.show()
