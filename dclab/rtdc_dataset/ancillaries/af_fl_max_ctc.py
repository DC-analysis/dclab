

from ... import features
from .ancillary_feature import AncillaryFeature


class MissingCrosstalkMatrixElementsError(BaseException):
    pass


def compute_ctc(mm, fl_channel):
    if "fl1_max" in mm:
        fl1 = mm["fl1_max"]
    else:
        fl1 = 0

    if "fl2_max" in mm:
        fl2 = mm["fl2_max"]
    else:
        fl2 = 0

    if "fl3_max" in mm:
        fl3 = mm["fl3_max"]
    else:
        fl3 = 0

    ctdict = {}

    for i in [1, 2, 3]:
        for j in [1, 2, 3]:
            if i == j:
                continue
            key = "crosstalk fl{}{}".format(i, j)
            par = "ct{}{}".format(i, j)
            if key in mm.config["calculation"]:
                ctdict[par] = mm.config["calculation"][key]

    if ("fl1_max" in mm and
        "fl2_max" in mm and
        "fl3_max" in mm and
        ("ct12" not in ctdict or
         "ct13" not in ctdict or
         "ct21" not in ctdict or
         "ct23" not in ctdict or
         "ct31" not in ctdict or
         "ct32" not in ctdict)):
        msg = "{}, has fl1_max, fl2_max, and fl3_max,".format(mm) \
              + " but not all crosstalk matrix elements are" \
              + " defined in the 'calculation' configuration section."
        raise MissingCrosstalkMatrixElementsError(msg)

    return features.fl_crosstalk.correct_crosstalk(
        fl1=fl1,
        fl2=fl2,
        fl3=fl3,
        fl_channel=fl_channel,
        **ctdict)


def compute_ctc1(mm):
    return compute_ctc(mm, fl_channel=1)


def compute_ctc2(mm):
    return compute_ctc(mm, fl_channel=2)


def compute_ctc3(mm):
    return compute_ctc(mm, fl_channel=3)


def get_method(fl_channel):
    if fl_channel == 1:
        return compute_ctc1
    elif fl_channel == 2:
        return compute_ctc2
    elif fl_channel == 3:
        return compute_ctc3


def register():
    opts_all = (["fl1_max",
                 "fl2_max",
                 "fl3_max"],
                ["crosstalk fl21",
                 "crosstalk fl31",
                 "crosstalk fl12",
                 "crosstalk fl32",
                 "crosstalk fl13",
                 "crosstalk fl23"])

    opts_12 = (["fl1_max",
                "fl2_max"],
               ["crosstalk fl21",
                "crosstalk fl12"])

    opts_13 = (["fl1_max",
                "fl3_max"],
               ["crosstalk fl31",
                "crosstalk fl13"])

    opts_23 = (["fl2_max",
                "fl3_max"],
               ["crosstalk fl32",
                "crosstalk fl23"])

    for flch in [1, 2, 3]:
        AncillaryFeature(feature_name="fl{}_max_ctc".format(flch),
                         method=get_method(flch),
                         req_features=opts_all[0],
                         req_config=[["calculation", opts_all[1]]],
                         priority=1)

    for flch in [1, 2]:
        AncillaryFeature(feature_name="fl{}_max_ctc".format(flch),
                         method=get_method(flch),
                         req_features=opts_12[0],
                         req_config=[["calculation", opts_12[1]]],
                         priority=0)

    for flch in [1, 3]:
        AncillaryFeature(feature_name="fl{}_max_ctc".format(flch),
                         method=get_method(flch),
                         req_features=opts_13[0],
                         req_config=[["calculation", opts_13[1]]],
                         priority=0)

    for flch in [2, 3]:
        AncillaryFeature(feature_name="fl{}_max_ctc".format(flch),
                         method=get_method(flch),
                         req_features=opts_23[0],
                         req_config=[["calculation", opts_23[1]]],
                         priority=0)
