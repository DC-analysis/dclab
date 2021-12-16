import copy

from .meta_parse import fbool, fint, fintlist, func_types, lcstr


#: All configuration keywords editable by the user
CFG_ANALYSIS = {
    # filtering parameters
    "filtering": [
        ["hierarchy parent", str, "Hierarchy parent of the dataset"],
        ["remove invalid events", fbool, "Remove events with inf/nan values"],
        ["enable filters", fbool, "Enable filtering"],
        ["limit events", fint, "Upper limit for number of filtered events"],
        ["polygon filters", fintlist, "Polygon filter indices"],
    ],
    # Addition user-defined data
    "calculation": [
        # "emodulus lut" was introduced in 0.32.0 and will replace
        # the deprecated "emodulus model".
        ["emodulus lut", str, "Look-up table identifier"],
        ["emodulus model", lcstr, "Model [DEPRECATED]"],
        ["emodulus medium", str, "Medium used (e.g. CellCarrierB, water)"],
        ["emodulus temperature", float, "Chip temperature [°C]"],
        ["emodulus viscosity", float, "Viscosity [Pa*s] if 'medium' unknown"],
        ["crosstalk fl21", float, "Fluorescence crosstalk, channel 2 to 1"],
        ["crosstalk fl31", float, "Fluorescence crosstalk, channel 3 to 1"],
        ["crosstalk fl12", float, "Fluorescence crosstalk, channel 1 to 2"],
        ["crosstalk fl32", float, "Fluorescence crosstalk, channel 3 to 2"],
        ["crosstalk fl13", float, "Fluorescence crosstalk, channel 1 to 3"],
        ["crosstalk fl23", float, "Fluorescence crosstalk, channel 2 to 3"],
    ]
}


#: All read-only configuration keywords for a measurement
CFG_METADATA = {
    # All parameters related to the actual experiment
    "experiment": [
        ["date", str, "Date of measurement ('YYYY-MM-DD')"],
        ["event count", fint, "Number of recorded events"],
        ["run index", fint, "Index of measurement run"],
        ["sample", str, "Measured sample or user-defined reference"],
        ["time", str, "Start time of measurement ('HH:MM:SS[.S]')"],
    ],
    # All special keywords related to RT-FDC
    # This section should not be present for regular RT-DC measurements.
    "fluorescence": [
        # The baseline offset was introduced in 0.33.0. It is added to
        # the trace data to obtain the actual signal used for data
        # processing (e.g. obtaining the fl1_max feature). This is more
        # robust than adding the offset directly to the trace data, because
        # of the possibility of integer overflows. Furthermore, DCKit can
        # set this parameter without modifying the original trace data
        # to correct/remove negative trace data
        # (see https://github.com/ZELLMECHANIK-DRESDEN/dclab/issues/101).
        # Note that traces accessed from RTDCBase instances are never
        # background-corrected!
        ["baseline 1 offset", fint, "Baseline offset channel 1"],
        ["baseline 2 offset", fint, "Baseline offset channel 2"],
        ["baseline 3 offset", fint, "Baseline offset channel 3"],
        ["bit depth", fint, "Trace bit depth"],
        # If a fluorescence channel is used, a channel name *must* be
        # present. If a channel is not used, the channel name *must not*
        # be present. E.g. if only channels 1 and 2 are used, but there
        # are three channels present, then `channel count` is two,
        # `channels installed` is three, and `channel 3 name` is not set.
        ["channel 1 name", str, "FL1 description"],
        ["channel 2 name", str, "FL2 description"],
        ["channel 3 name", str, "FL3 description"],
        ["channel count", fint, "Number of active channels"],
        ["channels installed", fint, "Number of available channels"],
        # In contrast to `channel ? name`, the laser power *may*
        # be present (but must be set to 0), if a laser line is not used.
        ["laser 1 lambda", float, "Laser 1 wavelength [nm]"],
        ["laser 1 power", float, "Laser 1 output power [%]"],
        ["laser 2 lambda", float, "Laser 2 wavelength [nm]"],
        ["laser 2 power", float, "Laser 2 output power [%]"],
        ["laser 3 lambda", float, "Laser 3 wavelength [nm]"],
        ["laser 3 power", float, "Laser 3 output power [%]"],
        ["laser count", fint, "Number of active lasers"],
        ["lasers installed", fint, "Number of available lasers"],
        ["sample rate", fint, "Trace sample rate [Hz]"],
        ["samples per event", fint, "Samples per event"],
        ["signal max", float, "Upper voltage detection limit [V]"],
        ["signal min", float, "Lower voltage detection limit [V]"],
        ["trace median", fint, "Rolling median filter size for traces"],
    ],
    # All tdms-related parameters
    "fmt_tdms": [
        ["video frame offset", fint, "Missing events at beginning of video"],
    ],
    # All imaging-related keywords
    "imaging": [
        ["flash device", str, "Light source device type"],  # e.g. green LED
        ["flash duration", float, "Light source flash duration [µs]"],
        ["frame rate", float, "Imaging frame rate [Hz]"],
        ["pixel size", float, "Pixel size [µm]"],
        ["roi position x", fint, "Image x coordinate on sensor [px]"],
        ["roi position y", fint, "Image y coordinate on sensor [px]"],
        ["roi size x", fint, "Image width [px]"],
        ["roi size y", fint, "Image height [px]"],
    ],
    # All parameters for online contour extraction from the event images
    "online_contour": [
        # The option "bg empty" was introduced in dclab 0.34.0 and
        # Shape-In 2.2.2.5.
        # Shape-In  writes to the "shapein-warning" log if there are
        # frames with event images (non-empty frames) that had to be
        # used for background correction.
        ["bg empty", fbool, "Background correction from empty frames only"],
        ["bin area min", fint, "Minium pixel area of binary image event"],
        ["bin kernel", fint, "Odd ellipse kernel size, binary image morphing"],
        ["bin threshold", fint, "Binary threshold for avg-bg-corrected image"],
        ["image blur", fint, "Odd sigma for Gaussian blur (21x21 kernel)"],
        ["no absdiff", fbool, "Avoid OpenCV 'absdiff' for avg-bg-correction"],
    ],
    # All online-filter-related keywords (box filters, soft limit, and
    # polygons are handled in `meta_logic`)
    "online_filter": [
        # "target*" is only set if measurement is stopped automatically.
        # "target*" is not necessarily reached (e.g. user aborted).
        ["target duration", float, "Target measurement duration [min]"],
        ["target event count", fint, "Target event count for online gating"],
    ],
    # All setup-related keywords, except imaging
    "setup": [
        ["channel width", float, "Width of microfluidic channel [µm]"],
        ["chip identifier", lcstr, "Unique identifier of the chip used"],
        ["chip region", lcstr, "Imaged chip region (channel or reservoir)"],
        ["flow rate", float, "Flow rate in channel [µL/s]"],
        ["flow rate sample", float, "Sample flow rate [µL/s]"],
        ["flow rate sheath", float, "Sheath flow rate [µL/s]"],
        ["identifier", str, "Unique setup identifier"],
        # "medium" is one of CellCarrier, CellCarrierB, water, or other
        ["medium", str, "Medium used"],
        ["module composition", str, "Comma-separated list of modules used"],
        ["software version", str, "Acquisition software with version"],
        ["temperature", float, "Mean chip temperature [°C]"],
    ],
}


# CFG convenience lists and dicts
_cfg = copy.deepcopy(CFG_METADATA)
_cfg.update(CFG_ANALYSIS)

#: dict with metadata description
config_descr = {}
for _key in _cfg:
    config_descr[_key] = {}
    for _subkey, __, _descr in _cfg[_key]:
        config_descr[_key][_subkey] = _descr

#: dict of dicts containing functions to convert input data
config_funcs = {}
for _key in _cfg:
    config_funcs[_key] = {}
    for _subkey, _type, __ in _cfg[_key]:
        config_funcs[_key][_subkey] = _type

#: dict with section as keys and config parameter names as values
config_keys = {}
for _key in _cfg:
    config_keys[_key] = [it[0] for it in _cfg[_key]]

#: dict of dicts containing the type of section parameters
config_types = {}
for _key in _cfg:
    config_types[_key] = {}
    for _subkey, _type, __ in _cfg[_key]:
        if _type in func_types:
            _type = func_types[_type]
        config_types[_key][_subkey] = _type
