"""Plugin Feature

This example shows how to create and use user-defined plugin features
in dclab.
The first downloadable file ("plugin_feature_script.py") in this section
contains the plugin, while the second file ("plugin_load_script.py")
contains code on how to load and utilise the plugin within dclab.

"""


def compute_some_new_features(rtdc_ds):
    """The function that does the heavy-lifting"""
    circ_per_area = rtdc_ds["circ"] / rtdc_ds["area_um"]
    circ_times_area = rtdc_ds["circ"] * rtdc_ds["area_um"]
    # returns a dictionary-like object
    return {"circ_per_area": circ_per_area, "circ_times_area": circ_times_area}


info = {
    "method": compute_some_new_features,
    "description": "This plugin will compute some features",
    "long description": "Even longer description that "
                        "can span multiple lines",
    "feature names": ["circ_per_area", "circ_times_area"],
    "feature labels": ["Circularity per Area", "Circularity times Area"],
    "features required": ["circ", "area_um"],
    "config required": [],
    "method check required": lambda x: True,
    "scalar feature": [True, True],
    "version": "0.1.0",
}
