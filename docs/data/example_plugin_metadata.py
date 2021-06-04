def compute_area_exponent(rtdc_ds):
    """Compute area^exp depending on the given user-defined metadata"""
    area_exp = rtdc_ds["area_um"] ** rtdc_ds.config["user"]["exp"]
    return {"area_exp": area_exp}


info = {
    "method": compute_area_exponent,
    "description": "This plugin computes area to the power of exp",
    "feature names": ["area_exp"],
    "features required": ["area_um"],
    "config required": [["user", ["exp"]]],
    "version": "0.1.0",
}
