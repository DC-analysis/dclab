"""Test access to private resources on GitHub"""
import copy
import os
import time

import dclab
from dclab.rtdc_dataset.fmt_dcor import api

from helper_methods import DCOR_AVAILABLE

import pytest

pytest.importorskip("requests")

if not DCOR_AVAILABLE:
    pytest.skip("No connection to DCOR", allow_module_level=True)


# 250209_Blood_2025-02-09_09.46_M003_Reference_dcn_export_28_minimal.rtdc
# a233aaf8-9998-4c44-8070-20fdba7cf3b2
rid = "aa452e24-6088-4bf7-839f-f7989ef38cf9"
access_token = os.environ.get("DCOR_API_TOKEN")


@pytest.mark.skipif(not access_token,
                    reason="DCOR_API_TOKEN environment variable not set")
def test_private_data_access():
    with dclab.new_dataset(rid, api_key=access_token) as ds:
        assert len(ds) == 28


def test_private_data_access_forbidden():
    """This is a control test without an API token"""
    with pytest.raises(api.DCORAccessError, match="Access denied"):
        dclab.new_dataset(rid)


@pytest.mark.skipif(not access_token,
                    reason="DCOR_API_TOKEN environment variable not set")
def test_basin_perishable_repr():
    with dclab.new_dataset(rid, api_key=access_token) as ds:
        bn = ds.basins[0]
        assert "valid" in repr(bn.perishable)


@pytest.mark.skipif(not access_token,
                    reason="DCOR_API_TOKEN environment variable not set")
def test_basin_refresh():
    with dclab.new_dataset(rid, api_key=access_token) as ds:
        ds.cache_basin_dict_time = 0.1
        bn = ds.basins[0]
        ds_bn0 = bn.ds
        assert bn.perishable
        assert not bn.perishable.perished()
        current_kwargs = copy.deepcopy(bn.perishable.expiration_kwargs)
        # Wait longer than `ds.cache_basin_dict_time` so it fetches the basin
        # information anew.
        time.sleep(0.2)
        # Refresh the basin.
        bn.perishable.refresh()
        assert bn.ds is not ds_bn0, "dataset should change"
        # Since the expiration time is computed using
        # "time_local_request" and there is random latency when
        # communicating with the server, the "time_local_expiration" in
        # the `expiration_kwargs` should always be different.
        assert current_kwargs != bn.perishable.expiration_kwargs


@pytest.mark.skipif(not access_token,
                    reason="DCOR_API_TOKEN environment variable not set")
def test_basin_as_dict():
    with dclab.new_dataset(rid, api_key=access_token) as ds:
        ds.cache_basin_dict_time = 0.1
        bn = ds.basins[0]
        bn_dict = bn.as_dict()
        assert bn_dict["perishable"]
        assert isinstance(bn_dict["perishable"], bool)
