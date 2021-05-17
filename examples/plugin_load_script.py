
import numpy as np
import dclab
import pathlib

# load a single plugin feature
dclab.load_plugin_feature("/path/to/plugin_example_features.py")

# loading multiple plugins
for plugin_path in pathlib.Path("my_plugin_directory").rglob("*.py"):
    dclab.load_plugin_feature(plugin_path)

# load some data
ds = dclab.new_dataset("path/to/rtdc/file")
# access the features
circ_per_area = ds["circ_per_area"]
circ_times_area = ds["circ_times_area"]

# do some filtering etc.
ds.config["filtering"]["circ_times_area min"] = 23
ds.config["filtering"]["circ_times_area max"] = 29
ds.apply_filter()
print("Removed {} out of {} events!".format(np.sum(~ds.filter.all), len(ds)))

# save the plugin features in a file
with dclab.new_dataset("/path/to/data.rtdc") as ds:
    # export the data to a new file
    ds.export.hdf5("/path/to/data_with_new_plugin_feature.rtdc",
                   features=ds.features_innate + ["circ_per_area",
                                                  "circ_times_area"])

# reload the plugin features at a later point
dclab.load_plugin_feature("/path/to/plugin.py")
ds = dclab.new_dataset("/path/to/data_with_new_plugin_feature.rtdc")
circ_per_area = ds["circ_per_area"]
circ_times_area = ds["circ_times_area"]
