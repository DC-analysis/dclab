
#: List of scalar (one scalar value per event) features. This
#: list does not include the `ml_score_???` features. If you
#: need find out whether a feature name is valid, please use
#: :func:`is_valid_feature`.
FEATURES_SCALAR = [
    ["area_cvx", "Convex area [px]"],
    # area_msd is the contour moment M00
    ["area_msd", "Measured area [px]"],
    ["area_ratio", "Porosity (convex to measured area ratio)"],
    # area_um is the convex area per definition
    ["area_um", "Area [µm²]"],
    ["aspect", "Aspect ratio of bounding box"],
    ["bright_avg", "Brightness average within contour [a.u.]"],
    ["bright_sd", "Brightness SD within contour [a.u.]"],
    ["circ", "Circularity"],
    # deform is computed from the convex contour
    ["deform", "Deformation"],
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
    ["frame", "Video frame number"],
    ["g_force", "Gravitational force in multiples of g"],
    # index starts with 1
    ["index", "Event index (Dataset)"],
    # index_online may have missing values (#71)
    ["index_online", "Event index (Online)"],
    # The inertia ratios of the event contours are defined by the
    # central second order moments of area (sqrt(m20/m02).
    ["inert_ratio_cvx", "Inertia ratio of convex contour"],
    ["inert_ratio_prnc", "Principal inertia ratio of raw contour"],
    ["inert_ratio_raw", "Inertia ratio of raw contour"],
    # This is an ancillary integer feature for visualizing the class
    # membership of individual events based on the `ml_score_???`
    # features.
    ["ml_class", "Most probable ML class"],
    ["nevents", "Total number of events in the same image"],
    ["pc1", "Principal component 1"],
    ["pc2", "Principal component 2"],
    # pos_x and pos_y are computed from the contour moments
    # "m10"/"m00" and "m01"/"m00" of the convex hull of "contour"
    ["pos_x", "Position along channel axis [µm]"],
    ["pos_y", "Position lateral in channel [µm]"],
    ["size_x", "Bounding box size x [µm]"],
    ["size_y", "Bounding box size y [µm]"],
    ["temp", "Chip temperature [°C]"],
    ["temp_amb", "Ambient temperature [°C]"],
    ["tilt", "Absolute tilt of raw contour"],
    ["time", "Event time [s]"],
    # Volume is computed from the raw contour (i.e. with exclusions).
    # Fun fact: If we had decided to compute it from the convex contour,
    # then we would have close to none pixelation effects ¯\_(ツ)_/¯.
    ["volume", "Volume [µm³]"],
]
# Add userdef features
for _i in range(10):
    FEATURES_SCALAR.append(["userdef{}".format(_i),
                            "User defined {}".format(_i)
                            ])

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
