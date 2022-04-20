"""DCOR-med access token (SSL certificate + CKAN token)"""
import pathlib
import ssl
import tempfile
import zipfile


def get_api_key(access_token_path, password):
    """Extract the API key / API token from an encrypted DCOR access token"""
    if isinstance(password, str):
        password = password.encode("utf-8")
    with zipfile.ZipFile(access_token_path) as arc:
        api_key = arc.read("api_key.txt", pwd=password).decode().strip()
    return api_key


def get_certificate(access_token_path, password):
    """Extract the certificate bundle from an encrypted DCOR access token"""
    if isinstance(password, str):
        password = password.encode("utf-8")
    with zipfile.ZipFile(access_token_path) as arc:
        cert_data = arc.read("server.cert", pwd=password)
    return cert_data


def get_hostname(access_token_path, password):
    """Extract the hostname from an encrypted DCOR access token"""
    cert_data = get_certificate(access_token_path, password)
    with tempfile.TemporaryDirectory(prefix="dcoraid_access_token_") as td:
        cfile = pathlib.Path(td) / "server.cert"
        cfile.write_bytes(cert_data)
        # Dear future-self,
        #
        # I know that this will probably not have been a good solution.
        # Anyway, I still decided to use this private function from the
        # built-in ssh module to avoid additional dependencies. Just so
        # you know: If you happen to be in trouble now because of this,
        # bear in mind that you had enough time to at least ask for the
        # functionality to be implemented in the requests library. Look
        # how I kept the lines all the same length!
        #
        # Cheers,
        # Paul
        cert_dict = ssl._ssl._test_decode_cert(str(cfile))
    # get the common name
    for ((key, value),) in cert_dict["subject"]:
        if key == "commonName":
            hostname = value.strip()
            break
    else:
        raise KeyError("Could not extract hostname from certificate!")
    return hostname
