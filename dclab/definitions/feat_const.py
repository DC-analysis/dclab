#: List of scalar (one scalar value per event) features. This
#: list does not include the `ml_score_???` features. If you
#: need find out whether a feature name is valid, please use
#: :func:`is_valid_feature`.
FEATURES_SCALAR = [
    ["area_cvx", "Convex area [px]"],
    # area_msd is the contour moment M00
    ["area_msd", "Measured area [px]"],
    ["area_ratio", "Porosity (convex to measured area ratio)"],
    # area_um is computed from the convex contour
    ["area_um", "Area [µm²]"],
    ["area_um_raw", "Area [µm²] of raw contour"],
    ["aspect", "Aspect ratio of bounding box"],
    # The background brightness of the frame (not of the mask)
    ["bg_med", "Median frame background brightness [a.u.]"],
    # Background image offset which should be added to "image_bg" before
    # performing background correction. This is used in cases where the
    # background stored in "image_bg" is not accurate enough, e.g.
    # because the image_bg is a median image for multiple events and
    # the imaging system exhibits flickering.
    ["bg_off", "Background offset [a.u.]"],
    # Brightness values are computed only for pixels inside the mask
    ["bright_avg", "Brightness average [a.u.]"],
    ["bright_sd", "Brightness SD [a.u.]"],
    ["bright_bc_avg", "Brightness average (bgc) [a.u.]"],
    ["bright_bc_sd", "Brightness SD (bgc) [a.u.]"],
    ["bright_perc_10", "10th Percentile of brightness (bgc)"],
    ["bright_perc_90", "90th Percentile of brightness (bgc)"],
    ["circ", "Circularity"],
    # deform is computed from the convex contour
    ["deform", "Deformation"],
    ["deform_raw", "Deformation of raw contour"],
    ["eccentr_prnc", "Eccentricity of raw contour"],
    ["emodulus", "Young's modulus [kPa]"],
    # fl*_area, fl*_pos, and fl*_width values correspond to the
    # object for which the contour was found. For high concentrations,
    # these values could be error-prone due to the assignment from
    # false objects.
    ["fl1_area", "FL-1 area of peak [a.u.]"],
    # fl1_dist is set to zero if there is only one peak
    ["fl1_dist", "FL-1 distance between two first peaks [µs]"],
    ["fl1_max", "FL-1 maximum [a.u.]"],
    ["fl1_max_ctc", "FL-1 maximum, crosstalk-corrected [a.u.]"],
    ["fl1_npeaks", "FL-1 number of peaks"],
    ["fl1_pos", "FL-1 position of peak [µs]"],
    ["fl1_width", "FL-1 width [µs]"],
    ["fl2_area", "FL-2 area of peak [a.u.]"],
    ["fl2_dist", "FL-2 distance between two first peaks [µs]"],
    ["fl2_max", "FL-2 maximum [a.u.]"],
    ["fl2_max_ctc", "FL-2 maximum, crosstalk-corrected [a.u.]"],
    ["fl2_npeaks", "FL-2 number of peaks"],
    ["fl2_pos", "FL-2 position of peak [µs]"],
    ["fl2_width", "FL-2 width [µs]"],
    ["fl3_area", "FL-3 area of peak [a.u.]"],
    ["fl3_dist", "FL-3 distance between two first peaks [µs]"],
    ["fl3_max", "FL-3 maximum [a.u.]"],
    ["fl3_max_ctc", "FL-3 maximum, crosstalk-corrected [a.u.]"],
    ["fl3_npeaks", "FL-3 number of peaks"],
    ["fl3_pos", "FL-3 position of peak [µs]"],
    ["fl3_width", "FL-3 width [µs]"],
    # Sum of the flow rates for sample and sheath flow
    ["flow_rate", "Flow rate [µLs⁻¹]"],
    ["frame", "Video frame number"],
    ["g_force", "Gravitational force in multiples of g"],
    # index starts with 1
    ["index", "Index (Dataset)"],
    # index_online enumerates the events found by Shape-In and may have
    # missing values in case of a writer-overrun with recovery (#71)
    ["index_online", "Index (Online)"],
    # The inertia ratios of the event contours are defined by the
    # central second order moments of area (sqrt(m20/m02).
    ["inert_ratio_cvx", "Inertia ratio of convex contour"],
    ["inert_ratio_prnc", "Principal inertia ratio of raw contour"],
    ["inert_ratio_raw", "Inertia ratio of raw contour"],
    # This is an ancillary integer feature for visualizing the class
    # membership of individual events based on the `ml_score_???`
    # features.
    ["ml_class", "Most probable ML class"],
    ["nevents", "Number of events in the same image"],
    ["pc1", "Principal component 1"],
    ["pc2", "Principal component 2"],
    # Convexity is defined by the ratio of convex contour length
    # versus raw contour length. We use inverse convexity as it is
    # more convenient to work with numbers >1.
    ["per_ratio", "Inverse Convexity (raw to convex perimeter ratio)"],
    ["per_um_raw", "Perimeter [µm] of raw contour"],
    # pos_x and pos_y are computed from the contour moments
    # "m10"/"m00" and "m01"/"m00" of the convex hull of "contour"
    ["pos_x", "Position along channel axis [µm]"],
    ["pos_y", "Position lateral in channel [µm]"],
    # Sum of the pressures applied to sample and sheath flow
    ["pressure", "Pressure [mPa]"],
    # QPI features computed from holographic data
    ["qpi_dm_avg", "Dry mass (average) [pg]"],
    ["qpi_dm_sd", "Dry mass (SD) [pg]"],
    ["qpi_pha_int", "Integrated phase [rad]"],
    ["qpi_ri_avg", "Refractive index (average)"],
    ["qpi_ri_sd", "Refractive index (SD)"],
    # QPI features from refocused events
    ["qpi_focus", "Computed focus distance [µm]"],
    # Size features
    ["size_x", "Bounding box size x [µm]"],
    ["size_y", "Bounding box size y [µm]"],
    # Ratio between two halves of a mask
    ["sym_x", "Symmetry ratio left-right"],
    ["sym_y", "Symmetry ratio top-bottom"],
    ["temp", "Chip temperature [°C]"],
    ["temp_amb", "Ambient temperature [°C]"],
    # Haralick texture features can be computed using the mahotas package
    # from the background-corrected and masked image
    ["tex_asm_avg", "Texture angular second moment (avg)"],  # H1
    ["tex_asm_ptp", "Texture angular second moment (ptp)"],  # H1
    ["tex_con_avg", "Texture contrast (avg)"],  # H2
    ["tex_con_ptp", "Texture contrast (ptp)"],  # H2
    ["tex_cor_avg", "Texture correlation (avg)"],  # H3
    ["tex_cor_ptp", "Texture correlation (ptp)"],  # H3
    ["tex_den_avg", "Texture difference entropy (avg)"],  # 11
    ["tex_den_ptp", "Texture difference entropy (ptp)"],  # 11
    ["tex_ent_avg", "Texture entropy (avg)"],  # H9
    ["tex_ent_ptp", "Texture entropy (ptp)"],  # H9
    ["tex_f12_avg", "Texture First measure of correlation (avg)"],  # 12
    ["tex_f12_ptp", "Texture First measure of correlation (ptp)"],  # 12
    ["tex_f13_avg", "Texture Second measure of correlation (avg)"],  # 13
    ["tex_f13_ptp", "Texture Second measure of correlation (ptp)"],  # 13
    ["tex_idm_avg", "Texture inverse difference moment (avg)"],  # H5
    ["tex_idm_ptp", "Texture inverse difference moment (ptp)"],  # H5
    ["tex_sen_avg", "Texture sum entropy (avg)"],  # H8
    ["tex_sen_ptp", "Texture sum entropy (ptp)"],  # H8
    ["tex_sva_avg", "Texture sum variance (avg)"],  # H7
    ["tex_sva_ptp", "Texture sum variance (ptp)"],  # H7
    ["tex_var_avg", "Texture variance (avg)"],  # H4
    ["tex_var_ptp", "Texture variance (ptp)"],  # H4
    ["tilt", "Absolute tilt of raw contour"],
    ["time", "Time [s]"],
    # Volume is computed from the raw contour (i.e. with exclusions).
    # Fun fact: If we had decided to compute it from the convex contour,
    # then we would have close to none pixelation effects ¯\_(ツ)_/¯.
    ["volume", "Volume [µm³]"],
]

#: User-defined features: They are not reserved for anything specific
#: and can be used by the user for e.g. prototyping.
for _i in range(10):
    FEATURES_SCALAR.append([f"userdef{_i}", f"User-defined {_i}"])

#: Basin mapping features: These are used for datasets that are derived from a
#: subset of events of another dataset. For instance, if a dataset consists
#: of every second event from another dataset, the feature `basinmap1` would
#: consist of the integer array `[1, 3, 5, 7, ...]` (indexing starts at zero).
#: The `basinmap1` feature must then be referenced in the corresponding basin
#: definition. These features should not be presented explicitly to the
#: normal user (e.g. in Shape-Out) to avoid ambiguities, and they should
#: always be exported alongside basins that refer to them.
for _j in range(10):
    FEATURES_SCALAR.append([f"basinmap{_j}", f"Basin mapping {_j}"])

#: list of non-scalar features
FEATURES_NON_SCALAR = [
    # This is a (M, 2)-shaped array with integer contour coordinates
    ["contour", "Event contour"],
    ["image", "Gray scale event image"],
    ["image_bg", "Gray scale event background image"],
    # This is the contour with holes filled
    ["mask", "Binary mask labeling the event in the image"],
    # See FLUOR_TRACES for valid keys
    ["trace", "Dictionary of fluorescence traces"],
    # QPI experimental holographic data
    ["qpi_oah", "Off-axis hologram"],
    # QPI holographic background data (experimental or computed)
    ["qpi_oah_bg", "Off-axis hologram background"],
    # QPI features computed from holographic data
    ["qpi_pha", "Hologram phase image [rad]"],
    ["qpi_amp", "Hologram amplitude image"],
]

#: List of fluorescence traces
FLUOR_TRACES = [
    "fl1_median",
    "fl1_raw",
    "fl2_median",
    "fl2_raw",
    "fl3_median",
    "fl3_raw",
]


# FEATURE convenience lists and dicts

#: list of feature names
feature_names = [_cc[0] for _cc in FEATURES_SCALAR + FEATURES_NON_SCALAR]

#: list of feature labels (same order as :const:`feature_names`
feature_labels = [_cc[1] for _cc in FEATURES_SCALAR + FEATURES_NON_SCALAR]

#: dict for converting feature names to labels
feature_name2label = {}

for _cc in FEATURES_SCALAR + FEATURES_NON_SCALAR:
    feature_name2label[_cc[0]] = _cc[1]

#: list of scalar feature names
scalar_feature_names = [_cc[0] for _cc in FEATURES_SCALAR]
