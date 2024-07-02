"""Tests for mapped basins

A mapped basin is a basins that contains more events than the dataset that
is referring to the basin. A mapped basin must contain all events that are
defined in the referring dataset.
"""
import json
import sys
import uuid

import h5py
import numpy as np

import pytest

import dclab
from dclab import new_dataset, RTDCWriter
from dclab.rtdc_dataset.feat_basin import (
    BasinmapFeatureMissingError, basin_priority_sorted_key
)


from helper_methods import DCOR_AVAILABLE, retrieve_data


class recursionlimit:
    def __init__(self, limit):
        self.limit = limit

    def __enter__(self):
        self.old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.limit)

    def __exit__(self, type, value, tb):
        sys.setrecursionlimit(self.old_limit)


def test_basin_basic_inception():
    """Create a mapped basin of a mapped basin"""
    h5path = retrieve_data("fmt-hdf5_image-mask-blood_2021.zip")
    with h5py.File(h5path, "a") as h5:
        # delete circularity to avoid ancillary feature computation in this
        # test.
        del h5["events"]["circ"]

    h5path_l1 = h5path.with_name("level1.rtdc")
    basin_map1 = np.array([1, 7, 10, 14], dtype=np.uint64)
    h5path_l2 = h5path.with_name("level2.rtdc")
    basin_map2 = np.array([1, 2], dtype=np.uint64)

    # level 1
    with dclab.new_dataset(h5path) as ds0, dclab.RTDCWriter(h5path_l1) as hw1:
        hw1.store_metadata(ds0.config.as_dict(pop_filtering=True))
        hw1.store_basin(basin_name="level1",
                        basin_type="file",
                        basin_format="hdf5",
                        basin_locs=[h5path],
                        basin_map=basin_map1
                        )

    # level 2
    with dclab.new_dataset(h5path) as ds0, dclab.RTDCWriter(h5path_l2) as hw2:
        hw2.store_metadata(ds0.config.as_dict(pop_filtering=True))
        hw2.store_basin(basin_name="level2",
                        basin_type="file",
                        basin_format="hdf5",
                        basin_locs=[h5path_l1],
                        # Explicitly name the basin here, otherwise the
                        # writer will override `basinmap0` with no `basinmap0`
                        # defined in h5path_l2.
                        basin_map=("basinmap2", basin_map2)
                        )

    # Checks for level 1
    with dclab.new_dataset(h5path) as ds0, dclab.new_dataset(h5path_l1) as ds1:
        assert np.all(ds1["basinmap0"] == basin_map1)
        assert len(ds1.basins) == 1
        assert ds1.basins[0].verify_basin()
        assert "mapped" in str(ds1.basins[0])
        assert "deform" in ds1.basins[0].features
        assert np.all(ds1["deform"] == ds0["deform"][basin_map1])

    # Checks for level 2
    with dclab.new_dataset(h5path) as ds0, dclab.new_dataset(h5path_l2) as ds2:
        assert np.all(ds2["basinmap2"] == basin_map2)
        assert len(ds2.basins) == 1
        assert "deform" in ds2.basins[0].features
        assert np.all(ds2["deform"] == ds0["deform"][basin_map1][basin_map2])


def test_basin_inception():
    """Make sure an error is thrown when diving into a deep inception

    We have ten basinmap features defined in dclab. Nobody will probably
    ever exhaust them (hello future self from the past :wave:), but just
    in case this happens, an informative error should be raised.
    """
    h5path = retrieve_data("fmt-hdf5_image-mask-blood_2021.zip")
    h5path_incept = h5path.with_name("inception.rtdc")
    mapping_array = np.arange(10, dtype=np.uint64)

    with h5py.File(h5path) as h5:
        attrs = dict(h5.attrs)

    with dclab.RTDCWriter(h5path_incept) as hw:
        hw.h5file.attrs.update(attrs)
        # Fill up all basinmap features
        for ii in range(10):
            # create a file-based basin with mapped content
            hw.store_feature(f"basinmap{ii}", mapping_array)
        # attempt to store a basin with a different basin mapping
        with pytest.raises(ValueError,
                           match="You have exhausted the usage of mapped "
                                 "basins"):
            hw.store_basin(
                basin_name="overload",
                basin_locs=[h5path],
                basin_format="hdf5",
                basin_type="file",
                basin_map=mapping_array + 1,  # has to be different
            )


@pytest.mark.parametrize("basinmap", [
    [1, 3, 4],  # very simple case
    [1, 1, 1, 2],  # not trivial, not realizable with hierarchy children
])
def test_basin_mapped(basinmap):
    path = retrieve_data("fmt-hdf5_image-mask-blood_2021.zip")
    with h5py.File(path, "a") as h5:
        # delete circularity to avoid ancillary feature computation in this
        # test.
        del h5["events"]["circ"]

    path_out = path.with_name("level1.rtdc")
    basinmap = np.array(basinmap, dtype=np.uint64)

    # create basin
    with dclab.new_dataset(path) as ds0, dclab.RTDCWriter(path_out) as hw1:
        hw1.store_metadata(ds0.config.as_dict(pop_filtering=True))
        hw1.store_basin(basin_name="level1",
                        basin_type="file",
                        basin_format="hdf5",
                        basin_locs=[path],
                        basin_map=basinmap
                        )

    # Checks basin
    with dclab.new_dataset(path) as ds0, dclab.new_dataset(path_out) as ds1:
        assert np.all(ds1["basinmap0"] == basinmap)
        assert len(ds1.basins) == 1
        assert ds1.basins[0].verify_basin()
        assert "mapped" in str(ds1.basins[0])
        assert "deform" in ds1.basins[0].features
        assert np.all(ds1["deform"] == ds0["deform"][basinmap])
        assert np.all(ds1["image"] == ds0["image"][:][basinmap])
        assert np.all(ds1["mask"] == ds0["mask"][:][basinmap])


@pytest.mark.parametrize("basinmap", [
    [1, 3, 4],  # very simple case
    [1, 1, 1, 2],  # not trivial, not realizable with hierarchy children
])
@pytest.mark.filterwarnings(
    "ignore::UserWarning")
def test_basin_mapped_export(basinmap):
    path = retrieve_data("fmt-hdf5_image-mask-blood_2021.zip")
    with h5py.File(path, "a") as h5:
        # delete circularity to avoid ancillary feature computation in this
        # test.
        del h5["events"]["circ"]

    path_out = path.with_name("level1.rtdc")
    basinmap = np.array(basinmap, dtype=np.uint64)

    # create basin
    with dclab.new_dataset(path) as ds0, dclab.RTDCWriter(path_out) as hw1:
        hw1.store_metadata(ds0.config.as_dict(pop_filtering=True))
        hw1.store_basin(basin_name="level1",
                        basin_type="file",
                        basin_format="hdf5",
                        basin_locs=[path],
                        basin_map=basinmap
                        )

    path_export = path.with_name("export.rtdc")
    with dclab.new_dataset(path_out) as ds1:
        ds1.filter.manual[:] = False
        ds1.filter.manual[:2] = True
        ds1.apply_filter()
        ds1.export.hdf5(
            path_export,
            features=["deform", "image", "mask"]
        )

    # Export data
    with dclab.new_dataset(path) as ds0, dclab.new_dataset(path_export) as ds1:
        assert np.all(ds1["deform"] == ds0["deform"][basinmap[:2]])
        assert np.all(ds1["image"] == ds0["image"][:][basinmap[:2]])
        assert np.all(ds1["mask"] == ds0["mask"][:][basinmap[:2]])


def test_error_when_basinmap_not_given():
    """This is a test for when the basinmap feature for mapping is missing

    A RecursionError is raised by dclab if the basinmap feature
    cannot be found in the data.
    """
    h5path = retrieve_data("fmt-hdf5_image-mask-blood_2021.zip")
    # create a file-based basin with mapped content
    h5path_small = h5path.with_name("smaller.rtdc")
    mapping_array = np.array([1, 7, 10, 14], dtype=np.uint64)

    # Dataset creation
    with h5py.File(h5path, "a") as src, RTDCWriter(h5path_small) as hw:
        # write experiment identifiers
        hw.h5file.attrs.update(src.attrs)
        unique_id = f"unique_id_{uuid.uuid4()}"
        src.attrs["experiment:run identifier"] = unique_id
        hw.h5file.attrs["experiment:run identifier"] = unique_id + "-hans"

        assert src["events/deform"].size == 18, "sanity check"
        assert "area_um" in src["events"]

        # first, copy parts of the scalar features to the new file
        hw.store_feature("deform", src["events/deform"][mapping_array])

        # Next, store the basin information in the new dataset
        bdat = {
            "type": "file",
            "format": "hdf5",
            "features": ["area_um"],
            "paths": [
                str(h5path),  # absolute path name
            ],
            "mapping": "basinmap0"  # note that this feature does not exist
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = hw.h5file.require_group("basins")
        hw.write_text(basins, "invalid_format_basin", blines)

    with dclab.new_dataset(h5path) as ds0, \
            dclab.new_dataset(h5path_small) as dss:
        # This is the feature that is in the small file
        assert np.all(ds0["deform"][mapping_array] == dss["deform"])
        # The area_um feature cannot be retrieved from the original file,
        # because the `basinmap0` feature is missing. Internally, dclab
        # locks up in a deep recursion involving ancillary features.
        with recursionlimit(100):  # speeds up this test
            with pytest.raises(BasinmapFeatureMissingError,
                               match="Could not find the feature 'basinmap0'"):
                dss["area_um"]


def test_export_to_hdf5_empty_features_mapped_basin():
    """Do not export any features, only the mapped basin information"""
    h5path = retrieve_data("fmt-hdf5_image-mask-blood_2021.zip")
    path_exp = h5path.with_name("exported.rtdc")
    with dclab.new_dataset(h5path) as ds:
        ds.filter.manual[:5] = False
        ds.apply_filter()
        ds.export.hdf5(
            path=path_exp,
            features=[],
            filtered=True,
            basins=True,
            )

    # Sanity check: Only the "deform" feature is in the dataset alongside
    # the basinmap feature, because this is a mapped basin.
    with h5py.File(path_exp) as h5:
        assert sorted(h5["events"].keys()) == ["basinmap0"]

    with dclab.new_dataset(path_exp) as dse:
        assert "deform" in dse
        assert "area_um" in dse
        assert np.allclose(dse["deform"][10], 0.057975024,
                           atol=1e-8, rtol=0)
        assert np.allclose(dse["area_um"][10], 79.0126,
                           atol=1e-3, rtol=0)
        assert "mapped" in str(dse.basins[0])


@pytest.mark.parametrize("filter1,filter2,offset", [
    # original dataset has "same" basins, filtering disabled
    [0, 0, 15],
    # original dataset has mapped basins, filtering disabled
    [10, 0, 5],
    # original dataset has "same" basins, filtering enabled
    [0, 10, 5],
    # original dataset has mapped basins, filtering enabled
    [5, 5, 5],
    [3, 7, 5],
])
def test_export_to_hdf5_inception(filter1, filter2, offset):
    """original dataset has "same" basins, filtering enabled"""
    h5path = retrieve_data("fmt-hdf5_image-mask-blood_2021.zip")
    path_exp1 = h5path.with_name("exported.rtdc")
    path_exp2 = h5path.with_name("exported_inception.rtdc")

    with dclab.new_dataset(h5path) as ds:
        nevents = len(ds)
        ds.filter.manual[:filter1] = False
        ds.apply_filter()
        ds.export.hdf5(
            path=path_exp1,
            features=["deform"],
            filtered=bool(filter1),
            basins=True,
        )

    with dclab.new_dataset(path_exp1) as ds1:
        # sanity check
        assert len(ds1) == nevents - filter1
        ds1.filter.manual[:filter2] = False
        ds1.apply_filter()
        ds1.export.hdf5(
            path=path_exp2,
            features=["deform"],
            filtered=bool(filter2),
            basins=True,
        )

    # Test presence of basinmap features
    with h5py.File(path_exp2) as h5:
        feature_list = ["deform"]
        if filter1 or filter2:
            feature_list.append("basinmap0")
        if filter1 and filter2:
            feature_list.append("basinmap1")
        assert sorted(h5["events"].keys()) == sorted(feature_list)

    with dclab.new_dataset(path_exp2) as ds2:
        # sanity check
        assert len(ds2) == nevents - filter1 - filter2
        assert "deform" in ds2
        assert "area_um" in ds2
        assert np.allclose(ds2["deform"][offset], 0.057975024,
                           atol=1e-8, rtol=0)
        assert np.allclose(ds2["area_um"][offset], 79.0126,
                           atol=1e-3, rtol=0)
        # cross-check with the original dataset
        with dclab.new_dataset(h5path) as ds0:
            fsum = filter1 + filter2
            assert np.allclose(ds2["deform"], ds0["deform"][fsum:])
            assert np.allclose(ds2["area_um"], ds0["area_um"][fsum:])


def test_export_to_hdf5_mapped_basin():
    """When exporting filtered data to HDF5, a mapped basin is added"""
    h5path = retrieve_data("fmt-hdf5_image-mask-blood_2021.zip")
    path_exp = h5path.with_name("exported.rtdc")
    with dclab.new_dataset(h5path) as ds:
        ds.filter.manual[:5] = False
        ds.apply_filter()
        ds.export.hdf5(
            path=path_exp,
            features=["deform"],
            filtered=True,
            basins=True,
            )

    # Sanity check: Only the "deform" feature is in the dataset alongside
    # the basinmap feature, because this is a mapped basin.
    with h5py.File(path_exp) as h5:
        assert sorted(h5["events"].keys()) == ["basinmap0", "deform"]

    with dclab.new_dataset(path_exp) as dse:
        assert "deform" in dse
        assert "area_um" in dse
        assert np.allclose(dse["deform"][10], 0.057975024,
                           atol=1e-8, rtol=0)
        assert np.allclose(dse["area_um"][10], 79.0126,
                           atol=1e-3, rtol=0)
        assert "mapped" in str(dse.basins[0])
        # cross-check with the original dataset
        with dclab.new_dataset(h5path) as ds0:
            assert np.allclose(dse["deform"], ds0["deform"][5:])
            assert np.allclose(dse["area_um"], ds0["area_um"][5:])


@pytest.mark.skipif(not DCOR_AVAILABLE, reason="DCOR not available")
@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_export_to_hdf5_from_dcor_dataset(tmp_path):
    """Exporting from DCOR with mapped basin should make basin available"""
    pytest.importorskip("requests")
    # TODO: Implement this test with a new, smaller DCOR dataset
    dcor_url = ("https://dcor.mpl.mpg.de/api/3/action/dcserv?id="
                "fb719fb2-bd9f-817a-7d70-f4002af916f0")
    path_exp = tmp_path / "exported_from_dcor.rtdc"
    with dclab.new_dataset(dcor_url) as ds:
        ds.filter.manual[5:] = False
        ds.apply_filter()
        ds.export.hdf5(
            path=path_exp,
            features=["deform"],
            filtered=True,
            basins=True,
            )

    with dclab.new_dataset(path_exp) as dse:
        # exported feature
        assert "deform" in dse
        assert "mapped" in str(dse.basins[0])
        assert np.allclose(dse["deform"][0], 0.009741939,
                           atol=1e-8, rtol=0)
        # basin feature
        assert dse["image"][2][10, 10] == 54


def test_export_to_hdf5_no_mapped_basin_filters_disabled():
    """When exporting data without filters to HDF5, a normal basin is added"""
    h5path = retrieve_data("fmt-hdf5_image-mask-blood_2021.zip")
    with h5py.File(h5path, "a") as h5:
        h5.attrs["experiment:run identifier"] = str(uuid.uuid4())

    path_exp = h5path.with_name("exported.rtdc")
    with dclab.new_dataset(h5path) as ds:
        ds.export.hdf5(
            path=path_exp,
            features=["deform"],
            filtered=False,
            basins=True,
            )

    # Sanity check: Only the "deform" feature is in the dataset.
    with h5py.File(path_exp) as h5:
        assert sorted(h5["events"].keys()) == ["deform"]

    with dclab.new_dataset(path_exp) as dse:
        assert "deform" in dse
        assert "area_um" in dse
        assert "mapped" not in str(dse.basins[0])
        assert np.allclose(dse["deform"][15], 0.057975024,
                           atol=1e-8, rtol=0)
        assert np.allclose(dse["area_um"][15], 79.0126,
                           atol=1e-3, rtol=0)

        # cross-check with the original dataset
        with dclab.new_dataset(h5path) as ds0:
            assert np.allclose(dse["deform"], ds0["deform"])
            assert np.allclose(dse["area_um"], ds0["area_um"])

    # And now a negative check with a modified basin identifier.
    with h5py.File(path_exp, "a") as h5e:
        h5e.attrs["experiment:run identifier"] = \
            h5e.attrs["experiment:run identifier"] + "-some-string"

    with dclab.new_dataset(path_exp) as dse2:
        assert "deform" in dse2
        assert "area_um" not in dse2  # sic
        assert np.allclose(dse2["deform"][15], 0.057975024,
                           atol=1e-8, rtol=0)


def test_sorting_priority_for_mapped_basins():
    """Basins are sorted by dclab

    Mapped basins should get lower priority.
    """
    mapper = "basinmap0"
    bnlist = [
        {"type": "remote", "format": "dcor", "mapping": "same", "ident": 0},
        {"type": "remote", "format": "dcor", "mapping": mapper, "ident": 1},
        {"type": "file", "format": "hdf5", "mapping": "same", "ident": 2},
        {"type": "file", "format": "hdf5", "mapping": mapper, "ident": 3},
        {"type": "hans", "format": "hdf5", "ident": 4},
        {"type": "remote", "format": "http", "ident": 5},
        {"type": "remote", "format": "http", "mapping": mapper, "ident": 6},

    ]
    sorted_list = sorted(bnlist, key=basin_priority_sorted_key)
    assert sorted_list[0]["ident"] == 2
    assert sorted_list[1]["ident"] == 3
    assert sorted_list[2]["ident"] == 5
    assert sorted_list[3]["ident"] == 6
    assert sorted_list[4]["ident"] == 0
    assert sorted_list[5]["ident"] == 1
    assert sorted_list[6]["ident"] == 4


def test_verify_basin_identifier():
    """The basin identifier for mapped basins must not match fully"""
    h5path = retrieve_data("fmt-hdf5_image-mask-blood_2021.zip")
    # create a file-based basin with mapped content
    h5path_small = h5path.with_name("smaller.rtdc")
    mapping_array = np.array([1, 7, 10, 14], dtype=np.uint64)

    # Dataset creation
    with h5py.File(h5path, "a") as src, RTDCWriter(h5path_small) as hw:
        # write experiment identifiers
        hw.h5file.attrs.update(src.attrs)
        unique_id = f"unique_id_{uuid.uuid4()}"
        src.attrs["experiment:run identifier"] = unique_id
        hw.h5file.attrs["experiment:run identifier"] = unique_id + "-hans"

        assert src["events/deform"].size == 18, "sanity check"
        assert "area_um" in src["events"]

        # first, copy parts of the scalar features to the new file
        hw.store_feature("deform", src["events/deform"][mapping_array])
        # mapping array
        hw.store_feature("basinmap0", mapping_array)

        # Next, store the basin information in the new dataset
        bdat = {
            "type": "file",
            "format": "hdf5",
            "features": ["area_um"],
            "paths": [
                str(h5path),  # absolute path name
            ],
            "mapping": "basinmap0"
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = hw.h5file.require_group("basins")
        hw.write_text(basins, "invalid_format_basin", blines)

    with new_dataset(h5path) as ds0, new_dataset(h5path_small) as dsb:
        assert len(dsb.basins) == 1
        assert np.all(ds0["area_um"][mapping_array] == dsb["area_um"])
        assert np.all(ds0["deform"][mapping_array] == dsb["deform"])


def test_writer_reuse_basinmap_feature():
    h5path = retrieve_data("fmt-hdf5_image-mask-blood_2021.zip")
    # create a file-based basin with mapped content
    h5path_small = h5path.with_name("smaller.rtdc")
    mapping_array = np.array([1, 7, 10, 14], dtype=np.uint64)

    with h5py.File(h5path, "a") as src, RTDCWriter(h5path_small) as hw:
        # write experiment identifiers
        hw.h5file.attrs.update(src.attrs)
        # store the basin once
        hw.store_basin(
            basin_name="exported with writer",
            basin_map=mapping_array,
            basin_locs=[h5path],
            basin_format="hdf5",
            basin_type="file",
            verify=False,
        )
        # store basin again, same mapping array
        hw.store_basin(
            basin_name="exported with writer",
            basin_map=mapping_array,
            basin_locs=["unknown.rtdc"],
            basin_format="hdf5",
            basin_type="file",
            verify=False,
        )

    # The essence of the test is to check that, even though two basins
    # were written, both use the same `mapping_array`.
    with h5py.File(h5path_small, "r") as h5:
        assert list(h5["events"].keys()) == ["basinmap0"]


def test_writer_verify_dataset_with_mapped_basin():
    h5path = retrieve_data("fmt-hdf5_image-mask-blood_2021.zip")
    # create a file-based basin with mapped content
    h5path_small = h5path.with_name("smaller.rtdc")
    mapping_array = np.array([1, 7, 10, 14], dtype=np.uint64)

    # Dataset creation
    with h5py.File(h5path, "a") as src, RTDCWriter(h5path_small) as hw:
        # write experiment identifiers
        hw.h5file.attrs.update(src.attrs)
        unique_id = f"unique_id_{uuid.uuid4()}"
        src.attrs["experiment:run identifier"] = unique_id
        hw.h5file.attrs["experiment:run identifier"] = unique_id + "-hans"

        assert src["events/deform"].size == 18, "sanity check"
        assert "area_um" in src["events"]

        # first, copy parts of the scalar features to the new file
        hw.store_feature("deform", src["events/deform"][mapping_array])

        # Write the mapped basin
        hw.store_basin(
            basin_name="exported with writer",
            basin_map=mapping_array,
            basin_locs=[h5path],
            basin_format="hdf5",
            basin_type="file",
            verify=True,  # this is the test
        )

    # Make sure this worked
    with dclab.new_dataset(h5path_small) as dse:
        assert "deform" in dse
        assert "area_um" in dse
        assert dse["image"][0][10, 10] == 73
