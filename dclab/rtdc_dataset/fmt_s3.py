import re


try:
    import s3fs
except ModuleNotFoundError:
    S3FS_AVAILABLE = False
else:
    S3FS_AVAILABLE = True


from .fmt_hdf5 import RTDC_HDF5


#: Regular expression for matching a DCOR resource URL
REGEXP_S3_URL = re.compile(
    r"^(https:\/\/)"  # protocol
    r"([a-z0-9-\.]*)"  # host
    r".*"  # path on host
)


class RTDC_S3(RTDC_HDF5):
    def __init__(self,
                 url: str,
                 secret_id: str = "",
                 secret_key: str = "",
                 *args, **kwargs):
        """Access RT-DC measurements in an S3-compatible object store

        This is essentially just a wrapper around :class:`.RTDC_HDF5`
        with `s3fs` passing a file object to h5py.

        Parameters
        ----------
        url: str
            Path to the object in an S3 instance
        secret_id: str
            S3 access identifier
        secret_key: str
            Secret S3 access key
        *args:
            Arguments for `RTDCBase`
        **kwargs:
            Keyword arguments for `RTDCBase`

        Attributes
        ----------
        path: pathlib.Path
            Path to the experimental HDF5 (.rtdc) file
        """
        if not S3FS_AVAILABLE:
            raise ModuleNotFoundError(
                "Package `s3fs` required for S3 format!")

        proto, s3_string = url.split("://", 1)
        s3_endpoint, s3_path = s3_string.split("/", 1)
        s3fskw = {
            "client_kwargs": {"endpoint_url": f"{proto}://{s3_endpoint}"},
            # A large block size makes loading metadata really slow.
            "default_block_size": 2048,
        }
        if secret_id and secret_key:
            # We have an id-key pair.
            s3fskw["key"] = secret_id
            s3fskw["secret"] = secret_key
            s3fskw["anon"] = False  # this is the default
        else:
            # Anonymous access has to be enabled explicitly.
            # Normally, s3fs would check for credentials in
            # environment variables and does not fall back to
            # anonymous use.
            s3fskw["anon"] = True

        self._fs = s3fs.S3FileSystem(**s3fskw)
        self._f3d = self._fs.open(s3_path, mode='rb')
        super(RTDC_S3, self).__init__(
            h5path=self._f3d,
            *args,
            **kwargs)
        # Override self.path with the actual S3 URL
        self.path = url


def is_s3_url(string):
    if not isinstance(string, str):
        return False
    else:
        return REGEXP_S3_URL.match(string.strip())
