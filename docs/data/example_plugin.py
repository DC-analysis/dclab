def compute_my_feature(rtdc_ds):
    """Compute circularity times area"""
    circ_times_area = rtdc_ds["circ"] * rtdc_ds["area_um"]
    return {"circ_times_area": circ_times_area}


info = {
    "method": compute_my_feature,
    "description": "This plugin computes area times circularity",
    "feature names": ["circ_times_area"],
    "features required": ["circ", "area_um"],
    "version": "0.1.0",
}
