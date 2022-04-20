import pathlib

import pytest

from dclab.rtdc_dataset import fmt_dcor
from dclab.rtdc_dataset.fmt_dcor import access_token

datapath = pathlib.Path(__file__).parent / "data"


def test_get_api_key():
    api_key = access_token.get_api_key(
        datapath / "example_access_token.dcor-access", "42")
    assert api_key == "7c0c7203-4e25-4b14-a118-553c496a7a52"


def test_get_certificate():
    cert = access_token.get_certificate(
        datapath / "example_access_token.dcor-access", "42").decode()
    assert "yTCCBLGgAwIBAgIUSrQD5LuXBSUtn41PeGDqP9XPbVIwDQYJKoZIhvcNA" in cert


def test_get_hostname():
    hostname = access_token.get_hostname(
        datapath / "example_access_token.dcor-access", "42")
    assert hostname == "dcor.example.com"


# feeling very confident about https://github.com/pytest-dev/pytest/pull/9871
@pytest.mark.xfail(pytest.version_tuple < (7, 2, 0),
                   reason="Requires pytest PR #9871 when run with coverage")
def test_store_and_get_certificate(tmp_path):
    cert = access_token.get_certificate(
        datapath / "example_access_token.dcor-access", "42")
    expect_path = tmp_path / "dcor.example.com.cert"
    expect_path.write_bytes(cert)
    fmt_dcor.DCOR_CERTS_SEARCH_PATHS.append(tmp_path)
    try:
        cert_path = fmt_dcor.get_server_cert_path("dcor.example.com")
    except BaseException:
        raise
    else:
        assert str(cert_path) == str(expect_path)
    finally:
        fmt_dcor.DCOR_CERTS_SEARCH_PATHS.clear()
