class IncompleteTDMSFileFormatError(BaseException):
    """Use for incomplete dataset (e.g. missing para.ini)"""


class InvalidTDMSFileFormatError(BaseException):
    """Use for invalid tdms files (e.g. unknown columns)"""
    pass
