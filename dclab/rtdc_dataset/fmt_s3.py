import functools
# import multiprocessing BaseManager here, because there is some kind
# of circular dependency issue with s3transfer.compat and multiprocessing.
from multiprocessing.managers import BaseManager  # noqa: F401
import os
import pathlib
import re
import socket
from urllib.parse import urlparse
import warnings


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
REGEXP_S3_BUCKET_KEY = re.compile(r"^[0-9a-z-]+(\/[0-9a-z-]+)+$")

S3_ENDPOINT_URL = os.environ.get("DCLAB_S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.environ.get("DCLAB_S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.environ.get("DCLAB_S3_SECRET_ACCESS_KEY")


class S3File(HTTPFile):
    """Monkeypatched `HTTPFile` to support authenticated access to S3"""
    def __init__(self,
                 object_path: str,
                 endpoint_url: str,
                 access_key_id: str = "",
                 secret_access_key: str = "",
                 use_ssl: bool = True,
                 verify_ssl: bool = True):
        """

        Parameters
        ----------
        object_path: str
            bucket/key path to object in the object store
        endpoint_url: str
            the explicit endpoint URL for accessing the object store
        access_key_id:
            S3 access key
        secret_access_key:
            secret S3 key mathcing `access_key_id`
        use_ssl: bool
            use SSL to connect to the endpoint, only disabled for testing
        verify_ssl: bool
            make sure the SSL certificate is sound, only used for testing
        """
        if endpoint_url is None:
            raise ValueError(
                "The S3 endpoint URL is empty. This could mean that you did "
                "not specify the full S3 URL or that you forgot to set "
                "the `S3_ENDPOINT_URL` environment variable.")
        endpoint_url = endpoint_url.strip().rstrip("/")
        self.botocore_session = botocore.session.get_session()
        self.s3_session = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            botocore_session=self.botocore_session)
        self.s3_client = self.s3_session.client(
            service_name='s3',
            use_ssl=use_ssl,
            verify=verify_ssl,
            endpoint_url=endpoint_url,
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
            endpoint_url=endpoint_url,
            config=config)

        bucket_name, object_name = object_path.strip("/").split("/", 1)
        self.s3_object = self.s3_resource.Object(
            bucket_name=bucket_name,
            key=object_name)

        super(S3File, self).__init__(f"{endpoint_url}/{object_path}")

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
                 endpoint_url: str = None,
                 access_key_id: str = None,
                 secret_access_key: str = None,
                 use_ssl: bool = True,
                 *args, **kwargs):
        """Access RT-DC measurements in an S3-compatible object store

        This is essentially just a wrapper around :class:`.RTDC_HDF5`
        with :mod:`boto3` and :class:`.HTTPFile` passing a file object to h5py.

        Parameters
        ----------
        url: str
            URL to an object in an S3 instance; this can be either a full
            URL (including the endpoint), or just `bucket/key`
        access_key_id: str
            S3 access identifier
        secret_access_key: str
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
                f"Package `boto3` required for loading S3 data '{url}'!")

        self._s3file = S3File(
            object_path=get_object_path(url),
            endpoint_url=(endpoint_url
                          or get_endpoint_url(url)
                          or S3_ENDPOINT_URL),
            access_key_id=(access_key_id
                           or S3_ACCESS_KEY_ID
                           or ""),
            secret_access_key=(secret_access_key
                               or S3_SECRET_ACCESS_KEY
                               or ""),
            use_ssl=use_ssl,
            verify_ssl=use_ssl,
            )
        # Initialize the HDF5 dataset
        super(RTDC_S3, self).__init__(
            h5path=self._s3file,
            *args,
            **kwargs)
        # Override self.path with the actual S3 URL
        self.path = self._s3file.url

    def close(self):
        super(RTDC_S3, self).close()
        self._s3file.close()


class S3Basin(Basin):
    basin_format = "s3"
    basin_type = "remote"

    def __init__(self, *args, **kwargs):
        self._available_verified = None
        super(S3Basin, self).__init__(*args, **kwargs)

    def _load_dataset(self, location, **kwargs):
        h5file = RTDC_S3(location, **kwargs)
        return h5file

    def is_available(self):
        """Check for boto3 and object availability

        Caching policy: Once this method returns True, it will always
        return True.
        """
        if self._available_verified is None:
            with self._av_check_lock:
                if not BOTO3_AVAILABLE:
                    self._available_verified = False
                else:
                    self._available_verified = \
                            is_s3_object_available(self.location)
        return self._available_verified


def is_s3_object_available(url: str,
                           access_key_id: str = None,
                           secret_access_key: str = None,
                           ):
    """Check whether an S3 object is available

    Parameters
    ----------
    url: str
        full URL to the object
    access_key_id: str
        S3 access identifier
    secret_access_key: str
        Secret S3 access key
    """
    avail = False
    if is_s3_url(url):
        endpoint_url = get_endpoint_url(url) or S3_ENDPOINT_URL
        if not endpoint_url:
            warnings.warn(
                f"Could not determine endpoint from URL '{url}'. Please "
                f"set the `S3_ENDPOINT_URL` environment variable or pass "
                f"a full object URL.")
        else:
            # default to https if no scheme or port is specified
            urlp = urlparse(endpoint_url)
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
                    s3file = S3File(
                        object_path=get_object_path(url),
                        endpoint_url=endpoint_url,
                        access_key_id=(access_key_id
                                       or S3_ACCESS_KEY_ID
                                       or ""),
                        secret_access_key=(secret_access_key
                                           or S3_SECRET_ACCESS_KEY
                                           or ""),
                        )
                    try:
                        s3file.s3_object.load()
                    except botocore.exceptions.ClientError:
                        avail = False
                    else:
                        avail = True
    return avail


@functools.lru_cache()
def get_endpoint_url(url):
    """Given a URL of an S3 object, return the endpoint URL

    Return None if no endpoint URL can be extracted (e.g. because
    just `bucket_name/object_path` was passed).
    """
    urlp = urlparse(url=url)
    if urlp.hostname:
        scheme = urlp.scheme or "https"
        port = urlp.port or (80 if scheme == "http" else 443)
        return f"{scheme}://{urlp.hostname}:{port}"
    else:
        return None


@functools.lru_cache()
def get_object_path(url):
    """Given a URL of an S3 object, return the `bucket_name/object_path` part

    Return object paths always without leading slash `/`.
    """
    urlp = urlparse(url=url)
    return urlp.path.lstrip("/")


@functools.lru_cache()
def is_s3_url(string):
    """Check whether `string` is a valid S3 URL using regexp"""
    if not isinstance(string, str):
        return False
    elif REGEXP_S3_URL.match(string.strip()):
        # this is pretty clear
        return True
    elif pathlib.Path(string).exists():
        # this is actually a file
        return False
    elif REGEXP_S3_BUCKET_KEY.match(string.strip()):
        # bucket_name/key
        return True
    else:
        return False
