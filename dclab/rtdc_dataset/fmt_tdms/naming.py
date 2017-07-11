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
   "inert_ratio": "InertiaRatio",
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
