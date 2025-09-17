import gc
import json
import time

import h5py
import numpy as np

import pytest

from dclab import rtdc_dataset, RTDCWriter, new_dataset
from dclab.rtdc_dataset.feat_basin import (
    Basin, PerishableRecord, get_basin_classes)
from dclab.rtdc_dataset.fmt_hdf5 import RTDC_HDF5

from helper_methods import retrieve_data


def make_perishable():
    """Return an HDF5 file that has a perishable basin defined"""
    # create a basin-based, perishable dataset
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, h5py.File(h5path_small, "w") as dst:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=dst,
                               features="scalar")
        # Next, store the basin information in the new dataset
        bdat = {
            "type": "file",
            "format": "hdf5perish",
            "paths": [
                h5path.name,  # relative path name
            ]
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = dst.require_group("basins")
        with RTDCWriter(dst, mode="append") as hw:
            hw.write_text(basins, "my_basin", blines)
        # sanity checks
        assert "deform" in dst["events"]
        assert "image" not in dst["events"]
    return h5path_small


@pytest.fixture(scope="function")
def with_perishable_basin():
    """Fixture for registering a perishable test basin type with dclab"""
    class PerishableBasinExample(Basin):
        basin_format = "hdf5perish"
        basin_type = "file"

        def __init__(self, *args, **kwargs):
            kwargs["perishable"] = self.set_perishable_record()
            super(PerishableBasinExample,  # noqa: F821
                  self).__init__(*args, **kwargs)

        @staticmethod
        def check_expired(basin, expiration_time):
            return expiration_time < time.monotonic()

        @staticmethod
        def refresh_basin(basin, extend_by=60):
            basin.set_perishable_record(time.monotonic() + extend_by)

        def set_perishable_record(self,
                                  expiration_time=time.monotonic() + 60,
                                  with_expiration=True,
                                  with_refresh=True,
                                  ):
            """Used in tests to set perishable"""
            kwargs = {}
            if with_expiration:
                kwargs["expiration_func"] = self.check_expired
                kwargs["expiration_kwargs"] = {
                    "expiration_time": expiration_time}
            if with_refresh:
                kwargs["refresh_func"] = self.refresh_basin
                kwargs["refresh_kwargs"] = {"extend_by": 60}
            self.perishable = PerishableRecord(self, **kwargs)
            return self.perishable

        def _load_dataset(self, location, **kwargs):
            return RTDC_HDF5(location, **kwargs)

        def is_available(self):
            return True

    assert "hdf5perish" in get_basin_classes()

    yield PerishableBasinExample

    # make this class undiscoverable in other tests
    del PerishableBasinExample.basin_format
    del PerishableBasinExample
    gc.collect()


def test_perish_basin_base(with_perishable_basin):
    """Instantiate perishable basin"""
    h5path_with_perishable_basin = make_perishable()

    # Now open the scalar dataset and check whether basins are defined
    with new_dataset(h5path_with_perishable_basin) as ds:
        assert "image" in ds.features_basin
        assert "image" not in ds.features_innate
        assert "image" in ds
        assert np.median(ds["image"][0]) == 151

        # basic basin checks
        assert len(ds.basins) == 1
        bn = ds.basins[0]
        assert bn.as_dict()["perishable"]
        assert not bn.perishable.perished()


def test_perish_basin_data_access_wo_expiration_func(with_perishable_basin):
    """When a perishable has no expiration_func, data access should work"""
    h5path_with_perishable_basin = make_perishable()

    # Now open the scalar dataset and check whether basins are defined
    with new_dataset(h5path_with_perishable_basin) as ds:
        # basic basin checks
        assert len(ds.basins) == 1
        bn = ds.basins[0]
        bn.set_perishable_record(with_expiration=False)
        assert bn.as_dict()["perishable"]
        assert not bn.perishable.perished()
        assert bn.perishable.perished() is None, "unknown expiration"
        # data access
        assert np.median(ds["image"][0]) == 151


def test_perish_basin_data_access_with_expiration_func(with_perishable_basin):
    """After the basins has perished, refresh should be called automatically"""
    h5path_with_perishable_basin = make_perishable()

    # Now open the scalar dataset and check whether basins are defined
    with new_dataset(h5path_with_perishable_basin) as ds:
        # basic basin checks
        assert len(ds.basins) == 1
        bn = ds.basins[0]
        assert bn.as_dict()["perishable"]
        bn.set_perishable_record(expiration_time=time.monotonic() - 5)
        # We have to test `perished` immediately after setting the record,
        # because calling `as_dict` will fetch all features which refreshes
        # the basin automatically.
        assert bn.perishable.perished()
        assert np.median(ds["image"][0]) == 151
        assert bn.as_dict()["perishable"]
        assert not bn.perishable.perished()


def test_perish_basin_ignored_on_export(with_perishable_basin):
    """Perishable basins should be ignored on export"""
    h5path_with_perishable_basin = make_perishable()
    export_path = h5path_with_perishable_basin.with_name("export.rtdc")

    # Now open the scalar dataset and check whether basins are defined
    with new_dataset(h5path_with_perishable_basin) as ds:
        ds.export.hdf5(export_path, features=["deform"], basins=True)

    with h5py.File(export_path) as h5:
        assert len(h5["basins"]) == 1, "only the smaller file is a basin"


def test_perish_basin_refresh_results_in_new_dataset(with_perishable_basin):
    h5path_with_perishable_basin = make_perishable()

    # Now open the scalar dataset and check whether basins are defined
    with new_dataset(h5path_with_perishable_basin) as ds:
        # basic basin checks
        assert len(ds.basins) == 1
        bn = ds.basins[0]
        ds0 = bn.ds
        ds1 = bn.ds
        assert ds0 is ds1
        assert not bn.perishable.perished()
        bn.set_perishable_record(expiration_time=time.monotonic() - 5)
        assert bn.perishable.perished()
        ds2 = bn.ds
        assert not bn.perishable.perished()
        assert ds2 is not ds1


def test_perish_basin_writing_warns(with_perishable_basin):
    """When explicitly writing perishable basins, there should be a warning"""
    h5path_with_perishable_basin = make_perishable()
    written = h5path_with_perishable_basin.with_name("written.rtdc")

    # Now open the scalar dataset and check whether basins are defined
    with new_dataset(h5path_with_perishable_basin) as ds:
        # basic basin checks
        assert len(ds.basins) == 1
        bn = ds.basins[0]

        with RTDCWriter(written) as hw:
            hw.store_metadata(ds.config.as_dict(pop_filtering=True))
            hw.store_feature("deform", ds["deform"])
            with pytest.warns(UserWarning, match="Storing perishable basin"):
                hw.store_basin(**bn.as_dict())

    with new_dataset(written) as ds:
        assert len(ds.basins) == 1


def test_perish_basin_write_without_expiration_func_should_write_true(
        with_perishable_basin):
    """perishable basin in json string should be True if no expiration_func"""
    h5path_with_perishable_basin = make_perishable()
    written = h5path_with_perishable_basin.with_name("written.rtdc")

    # Now open the scalar dataset and check whether basins are defined
    with new_dataset(h5path_with_perishable_basin) as ds:
        # basic basin checks
        assert len(ds.basins) == 1
        bn = ds.basins[0]
        bn.set_perishable_record(with_expiration=False)

        with RTDCWriter(written) as hw:
            hw.store_metadata(ds.config.as_dict(pop_filtering=True))
            hw.store_feature("deform", ds["deform"])
            with pytest.warns(UserWarning, match="Storing perishable basin"):
                hw.store_basin(**bn.as_dict())

    with new_dataset(written) as ds:
        assert len(ds.basins) == 1
        bndict = ds.basins_get_dicts()[0]
        assert bndict["perishable"]


def test_z_with_perishable_basin_fixture_does_not_leak(with_perishable_basin):
    """Make sure we are not globally registering the test basin"""
    "hdf5perish" not in get_basin_classes()
