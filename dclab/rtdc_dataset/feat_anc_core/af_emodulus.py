import warnings

from ... import features

from .ancillary_feature import AncillaryFeature


def compute_emodulus(mm):
    """Wrapper function for computing the Young's modulus

    Please take a look at the docs :ref:`sec_emodulus_usage`
    for more details on the three cases A, B, and C.
    There are also some sanity checks taking place here.
    """
    calccfg = mm.config["calculation"]

    medium = calccfg.get("emodulus medium", "other").lower()
    temperature = calccfg.get("emodulus temperature", None)
    viscosity = calccfg.get("emodulus viscosity", None)

    if viscosity is not None and medium == "other":
        # sanity checks
        if temperature is not None:
            warnings.warn("The 'emodulus temperature' configuration key is "
                          "ignored if the 'emodulus viscosity' key is set!")
        # Case B from the docs
        return compute_emodulus_visc_only(mm)
    else:
        # sanity checks
        if not isinstance(medium, str):
            raise ValueError(
                f"'emodulus medium' must be a string, got '{medium}'!")
        if medium not in features.emodulus.viscosity.KNOWN_MEDIA:
            raise ValueError(
                f"Only the following media are supported: "
                f"{features.emodulus.viscosity.KNOWN_MEDIA}, got '{medium}'!")
        if viscosity is not None:
            raise ValueError("You must not set the 'emodulus viscosity' "
                             "configuration keyword for known media!")
        # warnings
        if "emodulus viscosity model" not in calccfg:
            warnings.warn("Please specify the 'emodulus viscosity model' "
                          "key in the 'calculation' config segion, falling "
                          "back to 'herold-2017'!",
                          DeprecationWarning)
        # actual function calls
        if temperature is not None:
            # case C from the docs
            temperature = mm.config["calculation"]["emodulus temperature"]
            return compute_emodulus_known_media(mm, temperature=temperature)
        elif "temp" in mm:
            # case A from the docs
            return compute_emodulus_known_media(mm, temperature=mm["temp"])


def compute_emodulus_known_media(mm, temperature):
    """Only use known media and one temperature for all"""
    calccfg = mm.config["calculation"]
    # compute elastic modulus
    emod = features.emodulus.get_emodulus(
        area_um=mm["area_um"],
        deform=mm["deform"],
        medium=calccfg["emodulus medium"],
        channel_width=mm.config["setup"]["channel width"],
        flow_rate=mm.config["setup"]["flow rate"],
        px_um=mm.config["imaging"]["pixel size"],
        temperature=temperature,
        lut_data=calccfg["emodulus lut"],
        visc_model=calccfg.get("emodulus viscosity model", "herold-2017"),
    )
    return emod


def compute_emodulus_visc_only(mm):
    """The user entered the viscosity directly"""
    calccfg = mm.config["calculation"]
    # compute elastic modulus
    emod = features.emodulus.get_emodulus(
        area_um=mm["area_um"],
        deform=mm["deform"],
        medium=calccfg["emodulus viscosity"],
        channel_width=mm.config["setup"]["channel width"],
        flow_rate=mm.config["setup"]["flow rate"],
        px_um=mm.config["imaging"]["pixel size"],
        temperature=None,
        visc_model=None,
        lut_data=calccfg["emodulus lut"],
    )
    return emod


def is_channel(mm):
    """Check whether the measurement was performed in the channel

    If the chip region is not set, then it is assumed to be a
    channel measurement (for backwards compatibility and user-
    friendliness).
    """
    if "setup" in mm.config and "chip region" in mm.config["setup"]:
        region = mm.config["setup"]["chip region"]
        if region == "channel":
            # measured in the channel
            return True
        else:
            # measured in the reservoir
            return False
    else:
        # This might be a testing dictionary or someone who is
        # playing around with data. Avoid disappointments here.
        return True


def register():
    # Please note that registering these things is a delicate business,
    # because the priority has to be chosen carefully.
    # Note that here we have not included the "emodulus viscosity model"
    # configuration keyword. This is checked in the `compute_emodulus`
    # method above and a deprecation warning is issued, so old code
    # does not break immediately.
    for pr, vm in [(1, ["emodulus viscosity model"]),
                   (0, [])  # this is deprecated and should be removed!
                   ]:
        AncillaryFeature(feature_name="emodulus",
                         method=compute_emodulus,
                         data="case C",
                         req_features=["area_um", "deform"],
                         req_config=[["calculation", vm + [
                                        "emodulus lut",
                                        "emodulus medium",
                                        "emodulus temperature"]],
                                     ["imaging", ["pixel size"]],
                                     ["setup", ["flow rate", "channel width"]]
                                     ],
                         req_func=is_channel,
                         priority=4 + pr)
        AncillaryFeature(feature_name="emodulus",
                         data="case A",
                         method=compute_emodulus,
                         req_features=["area_um", "deform", "temp"],
                         req_config=[["calculation", vm + [
                                        "emodulus lut",
                                        "emodulus medium"]],
                                     ["imaging", ["pixel size"]],
                                     ["setup", ["flow rate", "channel width"]]
                                     ],
                         req_func=is_channel,
                         priority=0 + pr)

    AncillaryFeature(feature_name="emodulus",
                     data="case B",
                     method=compute_emodulus,
                     req_features=["area_um", "deform"],
                     req_config=[["calculation", vm + [
                                    "emodulus lut",
                                    "emodulus viscosity"]],
                                 ["imaging", ["pixel size"]],
                                 ["setup", ["flow rate", "channel width"]]
                                 ],
                     req_func=is_channel,
                     priority=2)
