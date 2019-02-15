# Keys are the feature names dclab and
# values are the feature names in the tdms file format.
dclab2tdms = {
    "area_cvx": "area",
    "area_msd": "raw area",
    "bright_avg": "Brightness",
    "bright_sd": "BrightnessSD",
    "circ": "circularity",
    "fl1_area": "FL1area",
    "fl1_dist": "FL1dpeaks",
    "fl1_max": "FL1max",
    "fl1_npeaks": "FL1npeaks",
    "fl1_pos": "FL1pos",
    "fl1_width": "FL1width",
    "fl2_area": "FL2area",
    "fl2_dist": "FL2dpeaks",
    "fl2_max": "FL2max",
    "fl2_npeaks": "FL2npeaks",
    "fl2_pos": "FL2pos",
    "fl2_width": "FL2width",
    "fl3_area": "FL3area",
    "fl3_dist": "FL3dpeaks",
    "fl3_max": "FL3max",
    "fl3_npeaks": "FL3npeaks",
    "fl3_pos": "FL3pos",
    "fl3_width": "FL3width",
    "g_force": "gValue",
    "frame": "time",  # [sic]
    "inert_ratio_cvx": "InertiaRatio",
    "inert_ratio_raw": "InertiaRatioRaw",
    "nevents": "NrOfCells",
    "pc1": "PC1",
    "pc2": "PC2",
    "pos_x": "x",
    "pos_y": "y",
    "size_x": "ax2",
    "size_y": "ax1",
    "temp": "temp",
    "temp_amb": "temp_amb",
}

# Add lower-case userdef features
for _i in range(10):
    dclab2tdms["userdef{}".format(_i)] = "userDef{}".format(_i)

# inverse of dclab2tdms
tdms2dclab = {}
for kk in dclab2tdms:
    tdms2dclab[dclab2tdms[kk]] = kk

# Add capitalized userdef features as well.
# see https://github.com/ZELLMECHANIK-DRESDEN/ShapeOut/issues/212
for _i in range(10):
    tdms2dclab["UserDef{}".format(_i)] = "userdef{}".format(_i)

# traces_tdms file definitions
# The second feature should not contain duplicates! - even if the
# entries in the first features are different.
tr_data_map = {
    "fl1_raw": ["fluorescence traces", "FL1raw"],
    "fl2_raw": ["fluorescence traces", "FL2raw"],
    "fl3_raw": ["fluorescence traces", "FL3raw"],
    "fl1_median": ["fluorescence traces", "FL1med"],
    "fl2_median": ["fluorescence traces", "FL2med"],
    "fl3_median": ["fluorescence traces", "FL3med"],
}

configmap = {
    "experiment": {
        "run index": ("General", "Measurement Number"),
        "sample": ("General", "Sample Name"),
        "date": ("General", "Date [YYYY-MM-DD]"),
        "time": ("General", "Start Time [hh:mm:ss]"),
    },
    # All special keywords related to RT-FDC
    "fluorescence": {
        "bit depth": ("FLUOR", "Bitdepthraw"),
        "channel count": ("FLUOR", "FL-Channels"),
        "channel 1 name": ("FLUOR", "Channel 1 Name"),
        "channel 2 name": ("FLUOR", "Channel 2 Name"),
        "channel 3 name": ("FLUOR", "Channel 3 Name"),
        "channels installed": ("FLUOR", "Channels Installed"),
        "laser 1 power": ("FLUOR", "Laser1 488 nm Power [%]"),
        "laser 2 power": ("FLUOR", "Laser2 561 nm Power [%]"),
        "laser 3 power": ("FLUOR", "Laser3 640 nm Power [%]"),
        "laser count": ("FLUOR", "Laser Count"),
        "lasers installed": ("FLUOR", "Lasers Installed"),
        "sample rate": ("FLUOR", "Samplerate [sps]"),
        "samples per event": ("FLUOR", "Samples Per Event"),
        "signal max": ("FLUOR", "ADCmax [V]"),
        "signal min": ("FLUOR", "ADCmin [V]"),
        "trace median": ("FLUOR", "Trace Median"),
    },
    # All tdms-related parameters
    "fmt_tdms": {
        "video frame offset": ("General", "video frame offset"),
    },
    # All imaging-related keywords
    "imaging": {
        "flash device": "LED (undefined)",
        "flash duration": ("General", "Shutter Time LED [us]"),
        "frame rate": ("Framerate", "Frame Rate"),
        "pixel size": ("Image", "Pix Size"),
        "roi position x": ("ROI", "x-pos"),
        "roi position y": ("ROI", "y-pos"),
        "roi size x": ("ROI", "width"),
        "roi size y": ("ROI", "height"),
    },
    # All parameters for online contour extraction from the event images
    "online_contour": {
        "bin area min": ("Image", "Trig Thresh"),
        "bin kernel": ("Image", "Bin Ops"),
        "bin threshold": ("Image", "Thresh"),
        "image blur": ("Image", "Blur"),
        "no absdiff": ("Image", "Diff_Method"),
    },
    # All online filters
    "online_filter": {
        "aspect min": ("Image", "Cell Aspect Min"),
        "aspect max": ("Image", "Cell Aspect Max"),
        "size_x max": ("Image", "Cell Max Length"),
        "size_y max": ("Image", "Cell Max Height"),
    },
    # All setup-related keywords, except imaging
    "setup": {
        "channel width": ("General", "Channel width [um]"),
        "chip region": ("General", "Region"),
        "flow rate": ("General", "Flow Rate [ul/s]"),
        "flow rate sample": ("General", "Sample Flow Rate [ul/s]"),
        "flow rate sheath": ("General", "Sheath Flow Rate [ul/s]"),
        "identifier": ("General", "Identifier"),
        "medium": ("General", "Buffer Medium"),
        "module composition": ("Image", "Setup"),
        "software version": ("General", "Software Version"),
    },
}
