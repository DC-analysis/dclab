class ContourIndexingError(BaseException):
    """Use when contour indices are confused"""


class IncompleteTDMSFileFormatError(BaseException):
    """Use for incomplete dataset (e.g. missing para.ini)"""


class InvalidTDMSFileFormatError(BaseException):
    """Use for invalid tdms files (e.g. unknown columns)"""


class InvalidVideoFileError(BaseException):
    """Used for bad video files"""


class CorruptFrameWarning(UserWarning):
    """Video frame corrupt or missing"""


class InitialFrameMissingWarning(CorruptFrameWarning):
    """Initial frame of video is missing"""


class SlowVideoWarning(UserWarning):
    """Getting video data will be slow"""


class MultipleSamplesPerEventFound(UserWarning):
    """Ambiguities in trace data"""
