"""RT-DC hdf5 format"""
from __future__ import annotations

from ...external.packaging import parse as parse_version


def get_software_version_from_h5(h5):
    software_version = h5.attrs.get("setup:software version", "")
    if isinstance(software_version, bytes):
        software_version = software_version.decode("utf-8")
    return software_version


def is_defective_feature_aspect(h5):
    """In Shape-In 2.0.6, there was a wrong variable cast"""
    software_version = get_software_version_from_h5(h5)
    return software_version in ["ShapeIn 2.0.6", "ShapeIn 2.0.7"]


def is_defective_feature_time(h5):
    """Shape-In stores the "time" feature as a low-precision float32

    This makes time resolution for large measurements useless,
    because times are only resolved with four digits after the
    decimal point. Here, we first check whether the "frame" feature
    and the [imaging]:"frame rate" configuration are set. If so,
    then we can compute "time" as an ancillary feature which will
    be more accurate than its float32 version.
    """
    # This is a necessary requirement. If we cannot compute the
    # ancillary feature, then we cannot ignore (even inaccurate) information.
    has_ancil = "frame" in h5["events"] and h5.attrs.get("imaging:frame rate",
                                                         0) != 0
    if not has_ancil:
        return False

    # If we have a 32 bit dataset, then things are pretty clear.
    is_32float = h5["events/time"].dtype.char[-1] == "f"
    if is_32float:
        return True

    # Consider the software
    software_version = get_software_version_from_h5(h5)

    # Only Shape-In stores false data, so we can ignore other recording
    # software.
    is_shapein = software_version.count("ShapeIn")
    if not is_shapein:
        return False

    # The tricky part: dclab might have analyzed the dataset recorded by
    # Shape-In, e.g. in a compression step. Since dclab appends its version
    # string to the software_version, we just have to parse that and make
    # sure that it is above 0.47.6.
    last_version = software_version.split("|")[-1].strip()
    if last_version.startswith("dclab"):
        dclab_version = last_version.split()[1]
        if parse_version(dclab_version) < parse_version("0.47.6"):
            # written with an older version of dclab
            return True

    # We covered all cases:
    # - ancillary information are available
    # - it's not a float32 dataset
    # - we excluded all non-Shape-In recording software
    # - it was not written with an older version of dclab
    return False


def is_defective_feature_volume(h5):
    """dclab computed volume wrong up until version 0.36.1"""
    # first check if the scripted fix was applied
    if "dclab_issue_141" in list(h5.get("logs", {}).keys()):
        return False
    # if that does not apply, check the software version
    software_version = get_software_version_from_h5(h5)
    if software_version:
        last_version = software_version.split("|")[-1].strip()
        if last_version.startswith("dclab"):
            dclab_version = last_version.split()[1]
            if parse_version(dclab_version) < parse_version("0.37.0"):
                return True
    return False


def is_defective_feature_inert_ratio(h5):
    """For long channels, there was an integer overflow until 0.48.1

    The problem here is that not only the channel length, but also
    the length of the contour play a role. All point coordinates of
    the contour are summed up and multiplied with one another which
    leads to integer overflows when computing mu20.

    Thus, this test is only a best guess, but still quite fast.

    See also https://github.com/DC-analysis/dclab/issues/212
    """
    # determine whether the image width is larger than 500px
    # If this file was written with dclab, then we always have the ROI size,
    # so we don't have to check the actual image.
    width_large = h5.attrs.get("imaging:roi size x", 0) > 500

    if width_large:
        # determine whether the software version was outdated
        software_version = get_software_version_from_h5(h5)
        if software_version:
            version_pipeline = [v.strip() for v in software_version.split("|")]
            last_version = version_pipeline[-1]
            if last_version.startswith("dclab"):
                dclab_version = last_version.split()[1]
                # The fix was implemented in 0.48.2, but this method here
                # was only implemented in 0.48.3, so we might have leaked
                # old data into new files.
                if parse_version(dclab_version) < parse_version("0.48.3"):
                    return True
    return False


def is_defective_feature_inert_ratio_raw_cvx(h5):
    """Additional check for `inert_ratio_raw` and `inert_ratio_cvx`

    These features were computed with Shape-In and were very likely
    computed correctly.

    See https://github.com/DC-analysis/dclab/issues/224
    """
    if is_defective_feature_inert_ratio(h5):
        # Possibly affected. Only return False if Shape-In check is negative
        software_version = get_software_version_from_h5(h5)
        version_pipeline = [v.strip() for v in software_version.split("|")]
        first_version = version_pipeline[0]
        if first_version.startswith("ShapeIn"):
            si_version = first_version.split()[1]
        elif "shapein-acquisition" in h5.get("logs", []):
            # Later versions of Shape-In do not anymore write "ShapeIn" in the
            # version string.
            si_version = first_version
        else:
            # Some other software was used to record the data and dclab
            # very likely stored the wrong inertia ratio.
            return True

        # We trust Shape-In >= 2.0.5
        if parse_version(si_version) >= parse_version("2.0.5"):
            return False

        return True

    return False


#: dictionary of defective features, defined by HDF5 attributes;
#: if a value matches the given HDF5 attribute, the feature is
#: considered defective
DEFECTIVE_FEATURES = {
    # feature: [HDF5_attribute, matching_value]
    "aspect": is_defective_feature_aspect,
    "inert_ratio_cvx": is_defective_feature_inert_ratio_raw_cvx,
    "inert_ratio_prnc": is_defective_feature_inert_ratio,
    "inert_ratio_raw": is_defective_feature_inert_ratio_raw_cvx,
    "tilt": is_defective_feature_inert_ratio,
    "time": is_defective_feature_time,
    "volume": is_defective_feature_volume,
}
