import functools
# import multiprocessing BaseManager here, because there is some kind
# of circular dependency issue with s3transfer.compat and multiprocessing.
from multiprocessing.managers import BaseManager  # noqa: F401
import re
import socket
from urllib.parse import urlparse


try:
    import boto3
    import botocore
    import botocore.client
    import botocore.exceptions
    import botocore.session
except ModuleNotFoundError:
    BOTO3_AVAILABLE = False
else:
    BOTO3_AVAILABLE = True

from ..http_utils import HTTPFile

from .feat_basin import Basin

from .fmt_hdf5 import RTDC_HDF5


#: Regular expression for matching a DCOR resource URL
REGEXP_S3_URL = re.compile(
    r"^(https?:\/\/)"  # protocol (http or https or omitted)
    r"([a-z0-9-\.]*)(\:[0-9]*)?\/"  # host:port
    r".+\/"  # bucket
    r".+"  # key
)


class S3File(HTTPFile):
    """Monkeypatched `HTTPFile` to support authenticated access to S3"""
    def __init__(self, url, access_key_id="", secret_access_key="",
                 use_ssl=True, verify_ssl=True):
        # Extract the bucket and object names
        s3_endpoint, s3_path = parse_s3_url(url)
        self.botocore_session = botocore.session.get_session()
        self.s3_session = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            botocore_session=self.botocore_session)
        self.s3_client = self.s3_session.client(
            service_name='s3',
            use_ssl=use_ssl,
            verify=verify_ssl,
            endpoint_url=s3_endpoint,
            )
        # Use a configuration that allows anonymous access
        # https://stackoverflow.com/a/34866092
        if not secret_access_key:
            config = botocore.client.Config(
                signature_version=botocore.UNSIGNED,
                region_name='us-east-1')
        else:
            config = None
        self.s3_resource = self.s3_session.resource(
            service_name="s3",
            use_ssl=use_ssl,
            verify=verify_ssl,
            endpoint_url=s3_endpoint,
            config=config)

        bucket_name, object_name = s3_path.strip("/").split("/", 1)
        self.s3_object = self.s3_resource.Object(
            bucket_name=bucket_name,
            key=object_name)
        super(S3File, self).__init__(url)

    def _parse_header(self):
        if self._len is None:
            self._len = self.s3_object.content_length
            self._etag = self.s3_object.e_tag

    def close(self):
        super(S3File, self).close()
        self.s3_client.close()

    def download_range(self, start, stop):
        """Download bytes given by the range (`start`, `stop`)

        `stop` is not inclusive (In the HTTP range request it normally is).
        """
        stream = self.s3_object.get(Range=f"bytes={start}-{stop-1}")['Body']
        return stream.read()


class RTDC_S3(RTDC_HDF5):
    def __init__(self,
                 url: str,
                 secret_id: str = "",
                 secret_key: str = "",
                 use_ssl: bool = True,
                 *args, **kwargs):
        """Access RT-DC measurements in an S3-compatible object store

        This is essentially just a wrapper around :class:`.RTDC_HDF5`
        with :mod:`boto3` and :class:`.HTTPFile` passing a file object to h5py.

        Parameters
        ----------
        url: str
            Full URL to an object in an S3 instance
        secret_id: str
            S3 access identifier
        secret_key: str
            Secret S3 access key
        use_ssl: bool
            Whether to enforce SSL (defaults to True)
        *args:
            Arguments for `RTDCBase`
        **kwargs:
            Keyword arguments for `RTDCBase`

        Attributes
        ----------
        path: str
            The URL to the object
        """
        if not BOTO3_AVAILABLE:
            raise ModuleNotFoundError(
                "Package `boto3` required for S3 format!")
        self._s3file = S3File(url,
                              access_key_id=secret_id,
                              secret_access_key=secret_key,
                              use_ssl=use_ssl,
                              verify_ssl=use_ssl,
                              )
        # Initialize the HDF5 dataset
        super(RTDC_S3, self).__init__(
            h5path=self._s3file,
            *args,
            **kwargs)
        # Override self.path with the actual S3 URL
        self.path = url

    def close(self):
        super(RTDC_S3, self).close()
        self._s3file.close()


class S3Basin(Basin):
    basin_format = "s3"
    basin_type = "remote"

    def __init__(self, *args, **kwargs):
        self._available_verified = None
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
        """Check for boto3 and object availability

        Caching policy: Once this method returns True, it will always
        return True.
        """
        with self._av_check_lock:
            if not BOTO3_AVAILABLE:
                self._available_verified = False
            if self._available_verified is None:
                self._available_verified = \
                    is_s3_object_available(self.location)
        return self._available_verified


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
                # Use `hostname`, not `netloc`, because `netloc` contains
                # the port number which we do not want here.
                s.connect((urlp.hostname, port))
            except (socket.gaierror, OSError):
                pass
            else:
                # Try to access the object
                s3file = S3File(url=url,
                                access_key_id=secret_id,
                                secret_access_key=secret_key)
                try:
                    s3file.s3_object.load()
                except botocore.exceptions.ClientError:
                    avail = False
                else:
                    avail = True
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
