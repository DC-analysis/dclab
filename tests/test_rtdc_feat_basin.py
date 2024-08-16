import json

import h5py
import numpy as np

import pytest

import dclab
from dclab import new_dataset, rtdc_dataset, RTDCWriter
from dclab.rtdc_dataset import feat_basin, fmt_http


from helper_methods import DCOR_AVAILABLE, retrieve_data


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.core.FeatureShouldExistButNotFoundWarning")
def test_basin_cyclic_dependency_found():
    """A basin can be defined in one of its sub-basins

    This is something dclab identifies and then
    raises a CyclicBasinDependencyFoundWarning
    """
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    paths_list = [
        h5path.with_name("level1.rtdc"),
        h5path.with_name("level2.rtdc"),
        h5path.with_name("level3.rtdc"),
    ]

    paths_list_rolled = np.roll(paths_list, 1)
    basins_list = []
    for pp in paths_list_rolled:
        basins_list.append({
            "basin_name": "user-defined basin",
            "basin_type": "file",
            "basin_format": "hdf5",
            "basin_feats": ["userdef1"],
            "basin_locs": [str(pp)],
            "verify": False,
        })

    for bdict, pp in zip(basins_list, paths_list):
        with h5py.File(h5path) as src, RTDCWriter(pp) as hw:
            # copy all the scalar features to the new file
            rtdc_dataset.rtdc_copy(src_h5file=src,
                                   dst_h5file=hw.h5file,
                                   features="scalar")
            hw.store_basin(**bdict)

    # Open the dataset
    with dclab.new_dataset(paths_list[0]) as ds:
        assert np.allclose(ds["deform"][0], 0.02494624), "sanity check"
        with pytest.warns(feat_basin.CyclicBasinDependencyFoundWarning,
                          match="Encountered cyclic basin dependency"):
            with pytest.raises(KeyError):
                _ = ds["userdef1"]


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.core.FeatureShouldExistButNotFoundWarning")
def test_basin_cyclic_dependency_found_2():
    """A basin can be defined in one of its sub-basins

    This is something dclab identifies and then
    raises a CyclicBasinDependencyFoundWarning
    """
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    # Those are the files with a cyclic dependency
    p1 = h5path.with_name("level1.rtdc")
    p2 = h5path.with_name("level2.rtdc")
    p3 = h5path.with_name("level3.rtdc")
    # This is the file that will contain the userdef1 feature
    pz = h5path.with_name("final.rtdc")

    # Initialize datasets
    for pp in [p1, p2, p3, pz]:
        with h5py.File(h5path) as src, RTDCWriter(pp) as hw:
            # copy all the scalar features to the new file
            rtdc_dataset.rtdc_copy(src_h5file=src,
                                   dst_h5file=hw.h5file,
                                   features="scalar")

    with RTDCWriter(pz) as hw:
        hw.store_feature("userdef1", hw.h5file["events/deform"][:])

    bn_kwargs = {
            "basin_type": "file",
            "basin_format": "hdf5",
            "basin_feats": ["userdef1"],
            "verify": False,
        }

    with RTDCWriter(p1) as hw:
        hw.store_basin(
            basin_name="link to path 2",
            basin_locs=[str(p2)],
            **bn_kwargs
        )

        # store a basin with a key that is sorted after the basin above
        bdat = {
            "name": "user-defined data",
            "type": "file",
            "format": "hdf5",
            "features": ["userdef1"],
            "paths": [str(pz)]
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = hw.h5file.require_group("basins")
        hw.write_text(basins, "zzzz99999zzzzz99999", blines)

    with RTDCWriter(p2) as hw:
        hw.store_basin(
            basin_name="link to path 3",
            basin_locs=[str(p3)],
            **bn_kwargs
        )

    with RTDCWriter(p3) as hw:
        hw.store_basin(
            basin_name="link to path 1, completing the circle",
            basin_locs=[str(p1)],
            **bn_kwargs
        )

    # Open the dataset
    with dclab.new_dataset(p1) as ds:
        assert np.allclose(ds["deform"][0], 0.02494624), "sanity check"
        assert "userdef1" in ds
        assert ds.basins[0].name == "link to path 2", "order matters"
        assert ds.basins[1].name == "user-defined data", "order matters"
        with pytest.warns(feat_basin.CyclicBasinDependencyFoundWarning,
                          match="Encountered cyclic basin dependency"):
            _ = ds["userdef1"]


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings(
    "ignore::UserWarning")
def test_basin_feature_image_export_from_basin_with_hierarchy(tmp_path):
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    epath = tmp_path / "exported.rtdc"
    epath2 = tmp_path / "exported2.rtdc"
    # Create a basins-only dataset using the export functionality
    with dclab.new_dataset(h5path) as ds:
        assert "image" in ds
        ds.export.hdf5(path=epath,
                       features=[],
                       basins=True,
                       filtered=False,
                       )
    # Attempt to export image data from the exported dataset
    with dclab.new_dataset(epath) as ds:
        ds2 = dclab.new_dataset(ds)
        assert "image" in ds2
        assert "image" not in ds2.features_innate
        ds2.export.hdf5(path=epath2,
                        features=["mask", "image"],
                        basins=False,
                        filtered=False,
                        )

    # Open the exported dataset and attempt to access the image feature
    with dclab.new_dataset(epath2) as ds:
        assert "mask" in ds
        assert np.any(ds["mask"][0])


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.skipif(not DCOR_AVAILABLE, reason="DCOR is not available")
@pytest.mark.parametrize("filtered", [True, False])
def test_basin_feature_nested_image_exported_available(tmp_path, filtered):
    pytest.importorskip("requests")
    epath = tmp_path / "exported.rtdc"
    # Load data from DCOR and export basin-only dataset
    with dclab.new_dataset("fb719fb2-bd9f-817a-7d70-f4002af916f0") as ds:
        assert "image" in ds
        ds.export.hdf5(path=epath,
                       features=[],
                       basins=True,
                       filtered=filtered,
                       )
    # Open the exported dataset and attempt to access the image feature
    with dclab.new_dataset(epath) as ds:
        assert "image" in ds
        assert np.any(ds["image"][0])
        imcount = 0
        for bn in ds.basins:
            if "image" in bn.ds:
                assert np.any(bn.ds["image"][0])
                imcount += 1
        assert imcount > 0, "image should be *some*-where"


def test_basin_hierarchy_trace():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")
        # Next, store the basin information in the new dataset
        bdat = {
            "type": "file",
            "format": "hdf5",
            "features": ["trace"],
            "paths": [
                str(h5path),  # absolute path name
            ]
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = hw.h5file.require_group("basins")
        hw.write_text(basins, "trace_basin", blines)

    ds = dclab.new_dataset(h5path_small)
    assert "trace" not in ds.features_innate
    ds2 = dclab.new_dataset(ds)
    assert "trace" in ds
    assert "trace" in ds2
    assert "fl1_raw" in ds["trace"]
    assert np.allclose(
        np.mean(ds2["trace"]["fl1_raw"][0]),
        24.5785536159601,
        atol=0, rtol=1e-5
    )


def test_basin_hierarchy_trace_missing():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")
        # Next, store the basin information in the new dataset
        bdat = {
            "type": "file",
            "format": "hdf5",
            "features": ["trace"],
            "paths": [
                str(h5path),  # absolute path name
            ]
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = hw.h5file.require_group("basins")
        hw.write_text(basins, "trace missing", blines)

    h5path.unlink()

    ds = dclab.new_dataset(h5path_small)
    ds2 = dclab.new_dataset(ds)
    ds2.apply_filter()
    assert "trace" not in ds2
    with pytest.raises(KeyError, match="does not contain the feature"):
        ds2["trace"]


@pytest.mark.skipif(not DCOR_AVAILABLE, reason="DCOR is not available")
def test_basin_not_allowed_to_have_local_basin_in_remote_basin():
    """Since version 0.57.5 we do forbid remote datasets with local basins

    This type of basin-inception would allow an attacker to access files
    on the local file system if the instance was passed on to him via a
    script or service. In addition, defining a local basin in a remote
    dataset does not have a sensible use case.
    """
    pytest.importorskip("requests")
    # sanity check
    s3_url = ("https://objectstore.hpccloud.mpcdf.mpg.de/"
              "circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/"
              "resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0")
    with fmt_http.RTDC_HTTP(s3_url) as dss3:
        assert not dss3._local_basins_allowed, "sanity check"

    # Because wo do not have the resources to upload anything and access
    # it via HTTP to test this, we simply modify the `_local_basins_allowed`
    # property of an `RTDC_HDF5` instance. This is a hacky workaround, but
    # it should be fine.

    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")
        # Next, store the basin information in the new dataset
        bdat = {
            "type": "file",
            "format": "hdf5",
            "features": ["image"],
            "paths": [
                str(h5path),  # absolute path name
            ]
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = hw.h5file.require_group("basins")
        hw.write_text(basins, "image_based_local_basin", blines)

    # sanity check
    with dclab.new_dataset(h5path_small) as ds:
        assert ds._local_basins_allowed, "sanity check"
        assert "image" in ds

    with dclab.new_dataset(h5path_small) as ds2:
        ds2._local_basins_allowed = False
        with pytest.warns(
            UserWarning,
                match="Basin type 'file' not allowed for format 'hdf5"):
            # Read the comments above carefully, if you wonder why this is.
            assert "image" not in ds2, "not there, local basins are forbidden"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.parametrize("path_sep", [r"/", r"\\"])
def test_basin_relative_paths(path_sep):
    """Relative paths should work for both `\\` and ¸`/`."""
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.parent / "subdirectory" / "relative.rtdc"
    h5path_small.parent.mkdir()

    with dclab.new_dataset(h5path) as ds0:
        assert "image" in ds0

    # Dataset creation
    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")
        # Next, store the basin information in the new dataset
        bdat = {
            "type": "file",
            "format": "hdf5",
            "features": ["image"],
            "paths": [f"..{path_sep}{h5path.name}"]
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = hw.h5file.require_group("basins")
        hw.write_text(basins, "relative_paths_basin", blines)

    with dclab.new_dataset(h5path_small) as ds:
        assert "image" not in ds.features_innate
        assert "image" in ds


def test_basin_sorting_basic():
    bnlist = [
        {"type": "remote", "format": "dcor", "ident": 0},
        {"type": "file", "format": "hdf5", "ident": 1},
        {"type": "hans", "format": "hdf5", "ident": 2},
        {"type": "remote", "format": "http", "ident": 3},
    ]
    sorted_list = sorted(bnlist, key=feat_basin.basin_priority_sorted_key)
    assert sorted_list[0]["ident"] == 1
    assert sorted_list[1]["ident"] == 3
    assert sorted_list[2]["ident"] == 0
    assert sorted_list[3]["ident"] == 2


@pytest.mark.parametrize("btype,bformat,sortval", [
    ["internal", "h5dataset", "aaa"],
    ["file", "hdf5", "bba"],
    ["remote", "http", "cca"],
    ["remote", "s3", "cda"],
    ["remote", "dcor", "cea"],
    ["peter", "hdf5", "zba"],
    ["remote", "hans", "cza"],
    ["hans", "peter", "zza"],
]
                         )
def test_basin_sorting_key(btype, bformat, sortval):
    bdict = {"type": btype,
             "format": bformat,
             }
    assert feat_basin.basin_priority_sorted_key(bdict) == sortval


def test_basin_unsupported_basin_format():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")
        # Next, store the basin information in the new dataset
        bdat = {
            "type": "file",
            "format": "peter",
            "paths": [
                "fake.rtdc",  # fake path
                str(h5path),  # absolute path name
            ]
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = hw.h5file.require_group("basins")
        hw.write_text(basins, "invalid_format_basin", blines)

    h5path.unlink()

    with pytest.warns(UserWarning,
                      match="Encountered unsupported basin format 'peter'"):
        with new_dataset(h5path_small) as ds:
            assert "image" not in ds
            assert not ds.features_basin


def test_basin_unsupported_basin_type():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")
        # Next, store the basin information in the new dataset
        bdat = {
            "type": "peter",
            "format": "hdf5",
            "paths": [
                "fake.rtdc",  # fake path
                str(h5path),  # absolute path name
            ]
        }
        blines = json.dumps(bdat, indent=2).split("\n")
        basins = hw.h5file.require_group("basins")
        hw.write_text(basins, "invalid_type_basin", blines)

    h5path.unlink()

    with pytest.warns(UserWarning,
                      match="Encountered unsupported basin type 'peter'"):
        with new_dataset(h5path_small) as ds:
            assert "image" not in ds
            assert not ds.features_basin
