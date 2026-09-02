import threading


class BasinFeatureMissingWarning(UserWarning):
    """Used when a badin feature is defined but not stored"""


class BasinIdentifierMismatchError(BaseException):
    """Used when the identifier of a basin does not match the definition"""


class CyclicBasinDependencyFoundWarning(UserWarning):
    """Used when a basin is defined in one of its sub-basins"""


class BasinmapFeatureMissingError(KeyError):
    """Used when one of the `basinmap` features is not defined"""


class BasinNotAvailableError(BaseException):
    """Used to identify situations where the basin data is not available"""


class BasinAvailabilityChecker(threading.Thread):
    """Helper thread for checking basin availability in the background"""
    def __init__(self, basin, *args, **kwargs):
        super().__init__(*args, daemon=True, **kwargs)
        self.basin = basin

    def run(self):
        self.basin.is_available()


def basin_priority_sorted_key(bdict: dict[str, str]):
    """Yield a sorting value for a given basin that can be used with `sorted`

    Basins are normally stored in random order in a dataset. This method
    brings them into correct order, prioritizing:

    - type: "file" over "remote"
    - format: "HTTP" over "S3" over "dcor"
    - mapping: "same" over anything else
    """
    srt_type = {
        "internal": "a",
        "file": "b",
        "remote": "c",
    }.get(bdict.get("type", "_"), "z")

    srt_format = {
        "h5dataset": "a",
        "hdf5": "b",
        "http": "c",
        "s3": "d",
        "dcor": "e",
    }.get(bdict.get("format", "_"), "z")

    mapping = bdict.get("mapping", "same")  # old dicts don't have "mapping"
    srt_map = "a" if mapping == "same" else mapping

    return srt_type + srt_format + srt_map
