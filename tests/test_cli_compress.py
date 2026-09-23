import hashlib
import json
import sys
import time

import dclab
from dclab import cli, new_dataset, rtdc_dataset, RTDCWriter
from dclab.rtdc_dataset.copier import ByteBookKeeper

import h5py
import numpy as np
import pytest

from helper_methods import retrieve_data


def test_check_suffix_disabled_compress():
    path_in_o = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    path_in = path_in_o.with_suffix("")
    path_in_o.rename(path_in)
    assert path_in.suffix == ""
    with pytest.raises(ValueError, match="Unsupported file type"):
        cli.compress(path_in=path_in,
                     path_out=path_in.with_name("compressed.rtdc"))
    # but this should work:
    cli.compress(path_in=path_in,
                 path_out=path_in.with_name("compressed2.rtdc"),
                 check_suffix=False)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_compress():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("compressed.rtdc")

    bbk = ByteBookKeeper()
    ret = cli.compress(path_in=path_in,
                       path_out=path_out,
                       byte_book_keeper=bbk,
                       )
    assert bbk.get_progress() == 1

    assert ret is None, "by default, this method should return 0 (exit 0)"
    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "dclab-compress" in dsj.logs
        assert len(dsj)
        assert len(dsj) == len(ds0)
        for feat in ds0.features:
            if feat in ["contour", "image", "mask"]:
                for ii in range(len(dsj)):
                    assert np.all(dsj[feat][ii] == ds0[feat][ii]), feat
            else:
                assert np.all(dsj[feat] == ds0[feat]), feat


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_compress_without_byte_book_keeper():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("compressed.rtdc")

    ret = cli.compress(path_in=path_in,
                       path_out=path_out,
                       )

    assert ret is None, "by default, this method should return 0 (exit 0)"
    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "dclab-compress" in dsj.logs
        assert len(dsj)
        assert len(dsj) == len(ds0)
        for feat in ds0.features:
            if feat in ["contour", "image", "mask"]:
                for ii in range(len(dsj)):
                    assert np.all(dsj[feat][ii] == ds0[feat][ii]), feat
            else:
                assert np.all(dsj[feat] == ds0[feat]), feat


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_compress_progress(capsys):
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("compressed.rtdc")

    bbk = ByteBookKeeper()
    cli.compress(path_in=path_in,
                 path_out=path_out,
                 byte_book_keeper=bbk,
                 )
    assert bbk.get_progress() == 1
    captured = capsys.readouterr()
    assert captured.out.count("Compression 100%")


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_compress_wo_logs():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    with h5py.File(path_in, "a") as h5:
        del h5["logs"]
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("compressed.rtdc")

    bbk = ByteBookKeeper()
    cli.compress(path_in=path_in,
                 path_out=path_out,
                 byte_book_keeper=bbk,
                 )
    assert bbk.get_progress() == 1

    with new_dataset(path_out) as ds:
        assert len(ds.logs) == 1


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_compress_already_compressed():
    """By default, an already compressed dataset should not be compressed"""
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out1 = path_in.with_name("compressed_1.rtdc")
    path_out2 = path_in.with_name("compressed_2.rtdc")
    path_out3 = path_in.with_name("compressed_copy_of_1.rtdc")
    # this is straight-forward
    cli.compress(path_out=path_out1, path_in=path_in)
    # just for the sake of comparison
    time.sleep(1)  # we need different time stamps in path_out2
    cli.compress(path_out=path_out2, path_in=path_in)
    # this is not trivial
    cli.compress(path_out=path_out3, path_in=path_out1)

    # the first two files should not be the same (dates are written, etc)
    h1 = hashlib.md5(path_out1.read_bytes()).hexdigest()
    h2 = hashlib.md5(path_out2.read_bytes()).hexdigest()
    h3 = hashlib.md5(path_out3.read_bytes()).hexdigest()
    assert h1 != h2
    # Changed in dclab 0.49.0: Since the compression step should also check
    # for defective features, it is important to revisit the entire file.
    # As such, it is cleaner to rewrite the entire dataset, since we can now
    # copy single HDF5 Datasets without having to redo the compression.
    # assert h1 == h3
    assert h1 != h3


def test_compress_basin_internal():
    """
    Internal basins should just be copied to the new file
    """
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")
    h5path_out = h5path.with_name("compressed.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")
        hw.store_basin(basin_name="example basin",
                       basin_type="internal",
                       basin_format="h5dataset",
                       basin_locs=["basin_events"],
                       basin_descr="an example test basin",
                       internal_data={"userdef1": np.arange(2)},
                       basin_map=np.zeros(src["events/deform"].shape[0]),
                       basin_feats=["userdef1"],
                       )

    # sanity check
    with new_dataset(h5path_small) as ds:
        assert "userdef1" in ds.features_basin
        assert "userdef1" in ds.features_innate

    # compress the basin-based dataset
    bbk = ByteBookKeeper()
    cli.compress(path_in=h5path_small,
                 path_out=h5path_out,
                 byte_book_keeper=bbk,
                 )
    assert bbk.get_progress() == 1

    with h5py.File(h5path_out) as h5:
        assert "deform" in h5["events"], "sanity check"
        assert "userdef1" not in h5["events"]
        assert "userdef1" in h5["basin_events"]
        assert np.all(h5["basin_events"]["userdef1"] == np.arange(2))

    with new_dataset(h5path_out) as ds:
        assert "userdef1" in ds.features_basin
        assert "userdef1" in ds.features_innate


def test_compress_basin_invalid_not_defined_at_all():
    """If a basin is specified that is not actually a basin
    """
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_2 = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")

    # Fake basin file
    with h5py.File(h5path, "a") as src, h5py.File(h5path_2, "a") as src2:
        src.attrs["experiment:run identifier"] = "hans"
        src2.attrs["experiment:run identifier"] = "hans"

    # sanity check
    with new_dataset(h5path) as ds:
        assert len(ds.basins) == 0

    with new_dataset(h5path_2) as ds:
        assert len(ds.basins) == 0

    path_out = h5path.with_name("out.rtdc")
    with pytest.raises(ValueError, match="not basins of the input file"):
        cli.compress(path_in=h5path_2,
                     path_out=path_out,
                     basin_input=[h5path]
                     )


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.feat_basin.CyclicBasinDependencyFoundWarning")
def test_compress_basin_invalid_measurement_identifier():
    """If a basin is specified that is not actually a basin
    """
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_2 = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")

    # Fake basin file
    with h5py.File(h5path, "a") as src, RTDCWriter(h5path_2) as hw:
        src.attrs["experiment:run identifier"] = "hans"
        hw.h5file.attrs["experiment:run identifier"] = "peter"  # HERE!
        del hw.h5file["events/area_um"]
        del hw.h5file["events/image"]
        hw.store_basin(basin_name="example basin",
                       basin_type="file",
                       basin_format="hdf5",
                       basin_locs=[str(h5path)],
                       basin_descr="an example test basin",
                       verify=False,  # don't verify here
                       )

    # sanity check
    with new_dataset(h5path) as ds:
        assert len(ds.basins) == 0
        assert "image" in ds

    with new_dataset(h5path_2) as ds:
        assert len(ds.basins) == 1
        assert "image" not in ds

    path_out = h5path.with_name("out.rtdc")
    with pytest.raises(ValueError, match="not basins of the input file"):
        cli.compress(path_in=h5path_2,
                     path_out=path_out,
                     basin_input=[h5path]
                     )


def test_compress_basin_valid_control():
    """If a basin is specified that is not actually a basin
    """
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_2 = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")

    # Fake basin file
    with h5py.File(h5path, "a") as src, RTDCWriter(h5path_2) as hw:
        src.attrs["experiment:run identifier"] = "hans"
        hw.h5file.attrs["experiment:run identifier"] = "hans"
        del hw.h5file["events/area_um"]
        del hw.h5file["events/image"]
        hw.store_basin(basin_name="example basin",
                       basin_type="file",
                       basin_format="hdf5",
                       basin_locs=[str(h5path)],
                       basin_descr="an example test basin",
                       verify=True,
                       )

    # sanity check
    with new_dataset(h5path) as ds:
        assert len(ds.basins) == 0
        assert "image" in ds

    with new_dataset(h5path_2) as ds:
        assert len(ds.basins) == 1
        assert "image" in ds

    path_out = h5path.with_name("out.rtdc")
    cli.compress(path_in=h5path_2,
                 path_out=path_out,
                 basin_input=[h5path]
                 )
    with new_dataset(path_out) as ds:
        assert "image" in ds.features_innate
        assert "area_um" in ds.features_innate


def test_compress_basin_no_data_from_basins():
    """
    When compressing a dataset, feature data from the basin should not be
    written to the output file.
    """
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")
    h5path_out = h5path.with_name("compressed.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, RTDCWriter(h5path_small) as hw:
        # first, copy all the scalar features to the new file
        rtdc_dataset.rtdc_copy(src_h5file=src,
                               dst_h5file=hw.h5file,
                               features="scalar")
        hw.store_basin(basin_name="example basin",
                       basin_type="file",
                       basin_format="hdf5",
                       basin_locs=[h5path],
                       basin_descr="an example test basin",
                       )

    # sanity check
    with new_dataset(h5path_small) as ds:
        assert "image" in ds.features

    # compress the basin-based dataset
    bbk = ByteBookKeeper()
    cli.compress(path_in=h5path_small,
                 path_out=h5path_out,
                 byte_book_keeper=bbk,
                 )
    assert bbk.get_progress() == 1

    with h5py.File(h5path_out) as h5:
        assert "deform" in h5["events"], "sanity check"
        assert "image" not in h5["events"], "Arrgh, basin feature was copied"

    with new_dataset(h5path_out) as ds:
        assert "image" in ds.features_basin


def test_compress_basin_preserved_compress():
    h5path = retrieve_data("fmt-hdf5_fl_wide-channel_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Empty, basin-based dataset creation
    with RTDCWriter(h5path_small) as hw:
        bn_hash = hw.store_basin(basin_name="example basin",
                                 basin_type="file",
                                 basin_format="hdf5",
                                 basin_locs=[h5path],
                                 basin_descr="an example test basin",
                                 )
    # compress the data
    h5path_out = h5path_small.with_name("compressed.rtdc")
    bbk = ByteBookKeeper()
    cli.compress(path_in=h5path_small,
                 path_out=h5path_out,
                 byte_book_keeper=bbk,
                 )
    assert bbk.get_progress() == 1

    with h5py.File(h5path_out) as h5_out, h5py.File(h5path_small) as h5_in:
        # check if h5path_out is empty
        if 'events' in h5_out:
            assert len(h5_out['events']) == 0
        # check if h5path_small basin is same as h5path_out basin
        assert 'basins' in h5_in
        assert 'basins' in h5_out
        assert np.all(
            h5_in['basins'][bn_hash][:] == h5_out['basins'][bn_hash][:]
        )

    with new_dataset(h5path) as ds, new_dataset(h5path_out) as ds_out:
        # check if all features of h5path are preserved in h5path_out
        assert len(ds.features_innate) == 32
        assert len(ds_out.features_innate) == 0
        for feat in ds.features_innate:
            assert feat in ds_out.features_basin


def test_compress_chopped():
    h5path = retrieve_data("fmt-hdf5_reference_2025.zip")
    path_out = h5path.with_name("compressed.rtdc")

    bbk = ByteBookKeeper()
    cli.compress(path_in=h5path,
                 path_out=path_out,
                 crop_event_images=True,
                 byte_book_keeper=bbk,
                 )
    assert bbk.get_progress() == 1

    with h5py.File(path_out) as h5:
        assert "image" in h5["basin_events"]
        assert "index" in h5["basin_events/image"]
        assert "0" in h5["basin_events/image"]
        # mask is not chopped

    with new_dataset(h5path) as ds0, new_dataset(path_out) as dsc:
        for ii in range(len(ds0)):
            assert np.all(ds0["mask"][ii] == dsc["mask"][ii])
            mask = ds0["mask"][ii]
            assert np.all(ds0["image"][ii][mask] == dsc["image"][ii][mask])


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_compress_log_md5_5m():
    """In dclab 0.42.0 we changed sha256 to md5-5M file checksums"""
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # sanity check (file is < 5MB)
    h1 = hashlib.md5(path_in.read_bytes()).hexdigest()
    assert h1 == "e49db02274ac75ab24911f893c41f5b0"
    # same directory (will be cleaned up with path_in)
    path_out1 = path_in.with_name("compressed_1.rtdc")

    bbk = ByteBookKeeper()
    cli.compress(path_out=path_out1,
                 path_in=path_in,
                 byte_book_keeper=bbk,
                 )
    assert bbk.get_progress() == 1

    with dclab.new_dataset(path_out1) as ds:
        log = ds.logs["dclab-compress"]
    dcdict = json.loads("\n".join(log))
    file = dcdict["files"][0]
    assert file["index"] == 1
    assert file["name"] == "mask_contour_reference.rtdc"
    assert file["md5-5M"] == "e49db02274ac75ab24911f893c41f5b0"


@pytest.mark.parametrize("basinmap", [
    [1, 3, 4],  # very simple case
    [1, 1, 1, 2],  # not trivial, not realizable with hierarchy children
])
def test_compress_with_basin_mapped(basinmap):
    path = retrieve_data("fmt-hdf5_image-mask-blood_2021.zip")
    with h5py.File(path, "a") as h5:
        # delete circularity to avoid ancillary feature computation in this
        # test.
        del h5["events"]["circ"]

    path_referrer = path.with_name("level1.rtdc")
    basinmap = np.array(basinmap, dtype=np.int64)

    # create metadata-only file that links to basin `path`
    with (dclab.new_dataset(path) as ds0,
          dclab.RTDCWriter(path_referrer) as hw1):
        hw1.store_metadata(ds0.config.as_dict(pop_filtering=True))
        hw1.store_basin(basin_name="level1",
                        basin_type="file",
                        basin_format="hdf5",
                        basin_locs=[path],
                        basin_map=basinmap
                        )

    # Compress with basins
    path_comp = path.with_name("compressed.rtdc")

    bbk = ByteBookKeeper()
    cli.compress(path_in=path_referrer,
                 path_out=path_comp,
                 basin_input=[path],
                 byte_book_keeper=bbk,
                 )
    assert bbk.get_progress() == 1

    # Checks compressed file
    with dclab.new_dataset(path) as ds0, dclab.new_dataset(path_comp) as ds1:
        assert np.all(ds1["basinmap0"] == basinmap)
        assert len(ds1.basins) == 1
        assert ds1.basins[0].verify_basin()
        assert "mapped" in str(ds1.basins[0])
        assert "deform" in ds1.basins[0].features
        assert np.all(ds1["deform"][:] == ds0["deform"][basinmap])
        assert np.all(ds1["image"][:] == ds0["image"][:][basinmap])
        assert np.all(ds1["mask"][:] == ds0["mask"][:][basinmap])


@pytest.mark.parametrize("basinmap", [
    [1, 3, 4],  # very simple case
    [1, 1, 1, 2],  # not trivial, not realizable with hierarchy children
])
def test_compress_with_basin_mapped_and_chopped(basinmap):
    path = retrieve_data("fmt-hdf5_reference_2025.zip")

    path_referrer = path.with_name("level1.rtdc")
    basinmap = np.array(basinmap, dtype=np.int64)

    # create metadata-only file that links to basin `path`
    with (dclab.new_dataset(path) as ds0,
          dclab.RTDCWriter(path_referrer) as hw1):
        hw1.store_metadata(ds0.config.as_dict(pop_filtering=True))
        hw1.store_basin(basin_name="level1",
                        basin_type="file",
                        basin_format="hdf5",
                        basin_locs=[path],
                        basin_map=basinmap
                        )

    # Compress with basins
    path_comp = path.with_name("compressed.rtdc")

    bbk = ByteBookKeeper()
    cli.compress(path_in=path_referrer,
                 path_out=path_comp,
                 basin_input=[path],
                 crop_event_images=True,
                 byte_book_keeper=bbk,
                 )
    assert bbk.get_progress() == 1

    with h5py.File(path_comp) as h5:
        assert "image" not in h5["events"]
        assert "image" in h5["basin_events"]

    # Check for chopped compression correctness
    with new_dataset(path_referrer) as ds0, new_dataset(path_comp) as dsc:
        assert len(ds0) == len(dsc)
        for ii in range(len(ds0)):
            assert np.all(ds0["mask"][ii] == dsc["mask"][ii])
            mask = ds0["mask"][ii]
            assert np.all(ds0["image"][ii][mask] == dsc["image"][ii][mask])

    # Check for basin correctness
    with dclab.new_dataset(path) as ds0, dclab.new_dataset(path_comp) as ds1:
        assert np.all(ds1["basinmap0"] == basinmap)
        assert len(ds1.basins) == 2

        for bn in ds1.basins:
            assert bn.verify_basin()
            if bn.basin_type == "internal" and bn.features == ["image"]:
                assert "image" in bn.features
                assert "chopped-image" in str(bn)
            elif bn.basin_type == "file" and bn.basin_format == "hdf5":
                assert "mapped" in str(bn)
                assert "area_um" in bn.features
                assert "deform" in bn.features
            else:
                assert False, "Only these two basins should be there"

        assert np.all(ds1["deform"][:] == ds0["deform"][basinmap])
        assert np.all(ds1["mask"][:] == ds0["mask"][:][basinmap])


def test_compress_with_logs():
    """Make sure only logs from the input file are stored"""
    path_basin = retrieve_data("fmt-hdf5_image-mask-blood_2021.zip")
    path_in = path_basin.with_name("input.rtdc")

    # Create downstream file with basin definition
    with dclab.new_dataset(path_basin) as ds:
        ds.export.hdf5(path=path_in,
                       features=["area_um", "deform", "mask"],
                       basins=True,
                       logs=True,
                       tables=True,
                       filtered=False)
    # Remove features in downstream file from basin file and add a new log
    with h5py.File(path_basin, "a") as h5:
        del h5["events/area_um"]
        del h5["events/deform"]
        del h5["events/mask"]
        with dclab.RTDCWriter(h5) as hw:
            hw.store_log("test-versteckt", ["Should", "not", "be", "there"])

    # Compress with basins
    path_comp = path_in.with_name("compressed.rtdc")

    bbk = ByteBookKeeper()
    cli.compress(path_in=path_in,
                 path_out=path_comp,
                 basin_input=[path_basin],
                 byte_book_keeper=bbk,
                 )
    assert bbk.get_progress() == 1

    # Checks compressed file
    with (
            dclab.new_dataset(path_in) as ds,
            dclab.new_dataset(path_basin) as dsb,
            dclab.new_dataset(path_comp) as dsc,
    ):
        for feat in ["area_um", "deform", "mask"]:
            assert feat in ds.features_innate
            assert feat not in ds.features_basin
            assert feat not in dsb.features_innate
            assert feat in dsc
            assert np.all(ds["area_um"][:] == dsc["area_um"][:])
            assert np.all(ds["deform"][:] == dsc["deform"][:])
            assert np.all(ds["mask"][:] == dsc["mask"][:])

        assert "test-versteckt" in dsb.logs
        assert "test-versteckt" not in ds.logs
        assert "test-versteckt" not in dsc.logs


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_compress_with_online_polygon_filters():
    """Shape-In 2.3 supports online polygon filters"""
    path = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # add an artificial online polygon filter
    with h5py.File(path, "a") as h5:
        # set soft filter to True
        h5.attrs["online_filter:area_um,deform soft limit"] = True
        # set filter values
        pf_name = "online_filter:area_um,deform polygon points"
        area_um = h5["events"]["area_um"]
        deform = h5["events"]["deform"]
        pf_points = np.array([
            [np.mean(area_um) + np.std(area_um),
             np.mean(deform)],
            [np.mean(area_um) + np.std(area_um),
             np.mean(deform) + np.std(deform)],
            [np.mean(area_um),
             np.mean(deform) + np.std(deform)],
        ])
        h5.attrs[pf_name] = pf_points

    path_out = path.with_name("compressed.rtdc")

    bbk = ByteBookKeeper()
    cli.compress(path_out=path_out,
                 path_in=path,
                 byte_book_keeper=bbk,
                 )
    assert bbk.get_progress() == 1

    with dclab.new_dataset(path_out) as ds:
        assert len(ds) == 8
        assert ds.config["online_filter"]["area_um,deform soft limit"]
        assert "area_um,deform polygon points" in ds.config["online_filter"]
        assert np.allclose(
            ds.config["online_filter"]["area_um,deform polygon points"],
            pf_points)


def test_compress_with_online_polygon_filters_real_data():
    """Shape-In 2.3 supports online polygon filters"""
    path = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")

    path_out = path.with_name("compressed.rtdc")

    bbk = ByteBookKeeper()
    cli.compress(path_out=path_out,
                 path_in=path,
                 byte_book_keeper=bbk,
                 )
    assert bbk.get_progress() == 1

    with dclab.new_dataset(path_out) as ds:
        assert len(ds) == 1
        assert ds.config["online_filter"]["size_x,size_y soft limit"]
        assert "size_x,size_y polygon points" in ds.config["online_filter"]
        assert np.allclose(
            ds.config["online_filter"]["size_x,size_y polygon points"],
            [[0.1, 0.2],
             [0.1, 2.5],
             [3.3, 3.2],
             [5.2, 0.9]]
        )


def test_version(capsys, monkeypatch):
    def sys_exit(status):
        return status
    monkeypatch.setattr(sys, "exit", sys_exit)
    monkeypatch.setattr(sys, "argv", ["dclab-compress", "--version"])

    parser = cli.compress_parser()
    parser.parse_args()

    output = capsys.readouterr().out.rstrip()
    assert output.count("dclab-compress")
    assert output.count(dclab.__version__)
    assert not output.count("usage")
