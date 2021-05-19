"""Load and use the Plugin Feature

This example shows how to use user-defined plugin features in dclab.
The first downloadable file ("plugin_feature_script.py") in the above section
contains the plugin, while this file ("plugin_load_script.py")
contains code on how to load and utilise the plugin within dclab.

"""


import matplotlib.pyplot as plt
import pathlib

import dclab

plugin_path = pathlib.Path(__file__).parent / "examples"

# load a single plugin feature
dclab.load_plugin_feature(plugin_path / "plugin_feature_script.py")

# load some data
ds = dclab.new_dataset("fb719fb2-bd9f-817a-7d70-f4002af916f0")
# access the features
circ_per_area = ds["circ_per_area"]
circ_times_area = ds["circ_times_area"]

# plot some features against eachother
xlabel = dclab.dfn.get_feature_label("deform")
ylabel = dclab.dfn.get_feature_label("circ_times_area")

fig = plt.figure(figsize=(6, 4))
ax1 = plt.subplot()
ax1.plot(ds["deform"], ds["circ_times_area"],
         "o", color="k", alpha=.2, ms=1)
ax1.set_xlabel(xlabel)
ax1.set_ylabel(ylabel)
ax1.set_xlim(0.0025, 0.025)
ax1.set_ylim(20, 40)
plt.show()
