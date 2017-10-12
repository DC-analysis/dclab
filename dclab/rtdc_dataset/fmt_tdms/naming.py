# Keys are the column names dclab and
# values are the column names in the tdms file format.
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
   "frame": "time", # [sic]
   "inert_ratio_cvx": "InertiaRatio",
   "inert_ratio_raw": "InertiaRatioRaw",
   "ncells": "NrOfCells",
   "pc1": "PC1",
   "pc2": "PC2",
   "pos_x": "x",
   "pos_y": "y",
   "size_x": "ax2",
   "size_y": "ax1",
   }

# Add userdef columns
for _i in range(10):
    dclab2tdms["userdef{}".format(_i)] = "userDef{}".format(_i)


# inverse of dclab2tdms
tdms2dclab = {}
for kk in dclab2tdms:
    tdms2dclab[dclab2tdms[kk]] = kk


# traces_tdms file definitions
# The second column should not contain duplicates! - even if the 
# entries in the first columns are different.
tr_data = [["fluorescence traces", "FL1raw"],
           ["fluorescence traces", "FL2raw"],
           ["fluorescence traces", "FL3raw"],
           ["fluorescence traces", "FL1med"],
           ["fluorescence traces", "FL2med"],
           ["fluorescence traces", "FL3med"],
        ]

configmap = {
    "experiment": {
        "run index": ("General", "Measurement Number"),
        },
    # All special keywords related to RT-FDC
    "fluorescence": {
        "bit depth": ("FLUOR", "Bitdepthraw"),
        "channel count": ("FLUOR", "FL-Channels"),
        "laser 1 power": ("FLUOR", "Laser Power 488 [mW]"),
        "laser 2 power": ("FLUOR", "Laser Power 561 [mW]"),
        "laser 3 power": ("FLUOR", "Laser Power 640 [mW]"),
        "laser 1 lambda": 488,
        "laser 2 lambda": 561,
        "laser 3 lambda": 640,
        "sample rate": ("FLUOR", "Samplerate"),
        "signal max": ("FLUOR", "ADCmax"),
        "signal min": ("FLUOR", "ADCmin"),
        },
    # All tdms-related parameters
    "fmt_tdms": {
        "video frame offset": ("General", "video frame offset"),
        },
    # All imaging-related keywords
    "imaging": {
        "exposure time": ("Framerate", "Shutter Time"),
        "flash current": ("General", "Current LED [A]"),
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
        "bin margin": ("Image", "Margin"),
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
        "channel width": ("General", "Channel width"),
        "chip region": ("General", "Region"),
        "flow rate": ("General", "Flow Rate [ul/s]"),
        "flow rate sample": ("General", "Sample Flow Rate [ul/s]"),
        "flow rate sheath": ("General", "Sheath Flow Rate [ul/s]"),
        #"medium": "CellCarrier",
        "module composition": ("Image", "Setup"),
        "software version": "tdms-acquisition (unknown)",
        "temperature": ("FLUOR", "Ambient Temperature"),
        #"viscosity": None,
        },
    }
