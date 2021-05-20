"""Working with plugin features

This example shows how to load a user-defined plugin feature recipe in dclab
and use it in a scatter plot.

Please also download the :download:`plugin_example.py
<../examples/plugin_example.py>` file for this example.
"""
import pathlib

import matplotlib.pyplot as plt

import dclab


plugin_path = pathlib.Path(__file__).parent

# load a single plugin feature
dclab.load_plugin_feature(plugin_path / "plugin_example.py")

# load some data from DCOR
ds = dclab.new_dataset("fb719fb2-bd9f-817a-7d70-f4002af916f0")

# access the features
circ_per_area = ds["circ_per_area"]
circ_times_area = ds["circ_times_area"]

# create a plot with a plugin feature
plt.figure(figsize=(8, 4))
xlabel = dclab.dfn.get_feature_label("circ_times_area")
ylabel = dclab.dfn.get_feature_label("deform")

ax1 = plt.subplot(title="Plot with a plugin feature")
ax1.plot(ds["circ_times_area"], ds["deform"],
         "o", color="k", alpha=.2, ms=1)
ax1.set_xlabel(xlabel)
ax1.set_ylabel(ylabel)
ax1.set_xlim(20, 40)
ax1.set_ylim(0.0025, 0.025)

plt.tight_layout()
plt.show()
