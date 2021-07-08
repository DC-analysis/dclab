class ContourIndexingError(BaseException):
    """Use when contour indices are confused"""
    pass


class IncompleteTDMSFileFormatError(BaseException):
    """Use for incomplete dataset (e.g. missing para.ini)"""


class InvalidTDMSFileFormatError(BaseException):
    """Use for invalid tdms files (e.g. unknown columns)"""
    pass


class InvalidVideoFileError(BaseException):
    """Used for bad video files"""
    pass


class CorruptFrameWarning(UserWarning):
    """Video frame corrupt or missing"""
    pass


class InitialFrameMissingWarning(CorruptFrameWarning):
    """Initial frame of video is missing"""
    pass


class SlowVideoWarning(UserWarning):
    """Getting video data will be slow"""
    pass


class MultipleSamplesPerEventFound(UserWarning):
    """Ambiguities in trace data"""
    pass
