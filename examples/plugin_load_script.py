import dclab

# load a single plugin feature
dclab.load_plugin_feature("/path/to/plugin_example_features.py")

# load some data
ds = dclab.new_dataset("path/to/rtdc/file")
# access the features
circ_per_area = ds["circ_per_area"]
circ_times_area = ds["circ_times_area"]

# do some filtering etc.
