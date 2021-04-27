
import numpy as np


def compute_fl_means(rtdc_ds):
    """The function that does the heavy-lifting"""
    fl1_mean = np.array([np.mean(
        rtdc_ds["trace"]["fl1_raw"][ii]) for ii in range(len(rtdc_ds))])
    fl2_mean = np.array([np.mean(
        rtdc_ds["trace"]["fl2_raw"][ii]) for ii in range(len(rtdc_ds))])
    # returns a dictionary-like object
    return {"fl1_mean": fl1_mean, "fl2_mean": fl2_mean}


info = {
    "method": compute_fl_means,
    "description": "This plugin will compute the mean of fl1_raw and fl2_raw",
    "feature names": ["fl1_mean", "fl2_mean"],
    "feature labels": ["", ""],
    "features required": ["area_cvx", "area_msd"],
    "config required": [],
    "method check required": "",
    "priority": 1,
    "version": "0.1.0",
}
