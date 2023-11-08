import functools
import re
import socket
from urllib.parse import urlparse


try:
    import s3fs
except ModuleNotFoundError:
    S3FS_AVAILABLE = False
else:
    S3FS_AVAILABLE = True


from .feat_basin import Basin

from .fmt_hdf5 import RTDC_HDF5


#: Regular expression for matching a DCOR resource URL
REGEXP_S3_URL = re.compile(
    r"^(https?:\/\/)"  # protocol (http or https or omitted)
    r"([a-z0-9-\.]*)(\:[0-9]*)?\/"  # host:port
    r".+\/"  # bucket
    r".+"  # key
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
            Full URL to an object in an S3 instance
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
        path: str
            The URL to the object
        """
        if not S3FS_AVAILABLE:
            raise ModuleNotFoundError(
                "Package `s3fs` required for S3 format!")
        s3fskw = get_s3fs_kwargs(url=url,
                                 secret_id=secret_id,
                                 secret_key=secret_key)
        _, s3_path = parse_s3_url(url)
        self._fs = s3fs.S3FileSystem(**s3fskw)
        self._f3d = self._fs.open(s3_path, mode='rb')
        # Initialize the HDF5 dataset
        super(RTDC_S3, self).__init__(
            h5path=self._f3d,
            *args,
            **kwargs)
        # Override self.path with the actual S3 URL
        self.path = url

    def close(self):
        super(RTDC_S3, self).close()
        self._f3d.close()


class S3Basin(Basin):
    basin_format = "s3"
    basin_type = "remote"

    def __init__(self, *args, **kwargs):
        self._available_verified = False
        super(S3Basin, self).__init__(*args, **kwargs)

    def load_dataset(self, location, **kwargs):
        h5file = RTDC_S3(location, enable_basins=False, **kwargs)
        # If the user specified the events of the basin, then store it
        # directly in the .H5Events class of .RTDC_HDF5. This saves us
        # approximately 2 seconds of listing the members of the "events"
        # group in the S3 object.
        h5file._events._features_list = self._features
        return h5file

    def is_available(self):
        """Check for s3fs and object availability

        Caching policy: Once this method returns True, it will always
        return True.
        """
        if not self._available_verified:
            self._available_verified = (
                    S3FS_AVAILABLE and is_s3_object_available(self.location))
        return self._available_verified


@functools.lru_cache()
def get_s3fs_kwargs(url: str,
                    secret_id: str = "",
                    secret_key: str = "",
                    ):
    """Return keyword arguments for s3fs

    Parameters
    ----------
    url: str
        full URL to the object
    secret_id: str
        S3 access identifier
    secret_key: str
        Secret S3 access key
    """
    s3_endpoint, _ = parse_s3_url(url)
    s3fskw = {
        "endpoint_url": s3_endpoint,
        # A large block size makes loading metadata really slow.
        "default_block_size": 2**18,
        # We are only ever opening one file per FS instance, so it does
        # not make sense to cache the instances.
        "skip_instance_cache": True,
        # We are not doing any FS directory listings
        "use_listings_cache": False,
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
    return s3fskw


def is_s3_object_available(url: str,
                           secret_id: str = "",
                           secret_key: str = "",
                           ):
    """Check whether an S3 object is available

    Parameters
    ----------
    url: str
        full URL to the object
    secret_id: str
        S3 access identifier
    secret_key: str
        Secret S3 access key
    """
    avail = False
    if is_s3_url(url):
        urlp = urlparse(url)
        # default to https if no scheme or port is specified
        port = urlp.port or (80 if urlp.scheme == "http" else 443)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            # Try to connect to the host
            try:
                s.connect((urlp.netloc, port))
            except (socket.gaierror, OSError):
                pass
            else:
                # Try to access the object
                s3fskw = get_s3fs_kwargs(url=url,
                                         secret_id=secret_id,
                                         secret_key=secret_key)
                _, s3_path = parse_s3_url(url)
                fs = s3fs.S3FileSystem(**s3fskw)
                try:
                    avail = fs.exists(s3_path)
                except OSError:
                    pass
    return avail


@functools.lru_cache()
def is_s3_url(string):
    """Check whether `string` is a valid S3 URL using regexp"""
    if not isinstance(string, str):
        return False
    else:
        return REGEXP_S3_URL.match(string.strip())


@functools.lru_cache()
def parse_s3_url(url):
    """Parse S3 `url`, returning `endpoint` URL and `key`"""
    urlp = urlparse(url)
    scheme = urlp.scheme or "https"
    port = urlp.port or (80 if scheme == "http" else 443)
    s3_endpoint = f"{scheme}://{urlp.hostname}:{port}"
    s3_path = urlp.path
    return s3_endpoint, s3_path
